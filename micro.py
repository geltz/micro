import sys
import os
import tempfile
import time
import math
import random
import numpy as np
import soundfile as sf
from scipy import signal

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QUrl, QTimer, QRectF, QPointF,
                          QPropertyAnimation, QEasingCurve, QSize)
from PyQt6.QtGui import (QColor, QPainter, QLinearGradient, QPen, QPainterPath, 
                         QBrush, QFont, QPixmap, QCursor, QPolygonF, QRadialGradient)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QFileDialog, QFrame, QGraphicsDropShadowEffect,
                             QMessageBox, QSizePolicy, QCheckBox)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- AUDIO ENGINE ---

class MicroEngine:
    @staticmethod
    def load_file(path):
        data, sr = sf.read(path, always_2d=False)
        if data.ndim > 1: data = data.mean(axis=1)
        return data.astype(np.float32), sr

    @staticmethod
    def save_file(path, data, sr):
        sf.write(path, data, sr)

    @staticmethod
    def apply_reverb(x, sr, mix=0.3):
        if mix <= 0.01: return x
        # Simple noise-based convolution reverb for "glue"
        tail_len = int(sr * 0.4) # Short room
        noise = np.random.randn(tail_len)
        env = np.exp(-np.linspace(0, 1, tail_len) * 7.0)
        ir = noise * env
        # Lowpass the tail slightly
        b, a = signal.butter(1, 0.4, 'low')
        ir = signal.lfilter(b, a, ir)
        
        wet = signal.fftconvolve(x, ir, mode='full')[:len(x)]
        wet = wet / (np.max(np.abs(wet)) + 1e-9)
        return (1 - mix) * x + mix * wet

    @staticmethod
    def get_zero_crossing(data, target_idx, search_window=1024):
        # Clamp bounds
        start = max(0, target_idx - search_window // 2)
        end = min(len(data), target_idx + search_window // 2)
        
        if start >= end: return target_idx
        
        chunk = data[start:end]
        
        # If stereo, sum the absolute values to find the quietest combined point
        if chunk.ndim > 1:
            # sum absolute amplitudes of channels
            amp_profile = np.sum(np.abs(chunk), axis=1)
        else:
            amp_profile = np.abs(chunk)
            
        # Find index of minimum amplitude
        min_local_idx = np.argmin(amp_profile)
        
        return start + min_local_idx

    @staticmethod
    def process_grain(data, sr, region, params):
        # 1. Calculate raw target indices
        raw_start = int(region[0] * len(data))
        raw_end = int(region[1] * len(data))
        
        # 2. Refine with Zero-Crossing Search (Snap to nearest quiet point)
        # Search +/- ~500 samples
        start_idx = MicroEngine.get_zero_crossing(data, raw_start, 1024)
        end_idx = MicroEngine.get_zero_crossing(data, raw_end, 1024)
        
        if start_idx >= end_idx: return np.zeros((1024, 2), dtype=np.float32)
        
        chunk = data[start_idx:end_idx].copy()
        if len(chunk) == 0: return np.zeros((1024, 2), dtype=np.float32)

        # 3. Safety De-Click (Micro-fades)
        # Apply a 2ms fade in/out to the raw cut to guarantee 0.0 edges
        # independent of the user's "Attack/Release" settings.
        fade_len = min(int(sr * 0.002), len(chunk) // 2) 
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            
            # Handle dimensions for broadcasting
            if chunk.ndim > 1:
                fade_in = fade_in[:, None]
                fade_out = fade_out[:, None]
                
            chunk[:fade_len] *= fade_in
            chunk[-fade_len:] *= fade_out

        # [Mono Processing Chain]
        
        if params.get('reverse', False): chunk = chunk[::-1]

        rate = params.get('rate', 1.0)
        if abs(rate - 1.0) > 0.01:
            new_len = int(len(chunk) / rate)
            if new_len > 0: chunk = signal.resample(chunk, new_len)

        # Soft Crush
        crush = params.get('crush', 0.0)
        if crush > 0.05:
            factor = 1.0 - (crush * 0.8) 
            target_len = int(len(chunk) * factor)
            if target_len > 10:
                down = signal.resample(chunk, target_len)
                chunk = signal.resample(down, len(chunk))

        # Tone
        tone = params.get('tone', 1.0)
        if tone < 0.98:
            cutoff = 200 * (100 ** tone)
            sos = signal.butter(2, cutoff, 'low', fs=sr, output='sos')
            chunk = signal.sosfilt(sos, chunk)
            # Drift
            t = np.linspace(0, len(chunk)/sr, len(chunk))
            mod = 1.0 + (0.15 * (1.0-tone)) * np.sin(2 * np.pi * 2.0 * t)
            if chunk.ndim > 1: mod = mod[:, None]
            chunk = chunk * mod * (1.0 + (1.0-tone)*0.4)

        # Compressor
        comp = params.get('compress', 0.0)
        if comp > 0.01:
            sos_env = signal.butter(1, 0.02, output='sos')
            # Detect on mono sum if stereo
            det_sig = np.mean(np.abs(chunk), axis=1) if chunk.ndim > 1 else np.abs(chunk)
            env = signal.sosfilt(sos_env, det_sig)
            
            thresh = 0.6 * (1.0 - comp * 0.7)
            gain_red = np.minimum(1.0, thresh / (env + 1e-6))
            
            if chunk.ndim > 1: gain_red = gain_red[:, None]
            chunk = chunk * gain_red * (1.0 + comp * 2.5)

        # User Envelopes (Musical ADSR)
        att_p = params.get('attack', 0.01)
        rel_p = params.get('release', 0.01)
        n_samples = len(chunk)
        att_s = int(n_samples * att_p)
        rel_s = int(n_samples * rel_p)
        
        if att_s + rel_s > n_samples:
            scale = n_samples / (att_s + rel_s + 1)
            att_s = int(att_s * scale)
            rel_s = int(rel_s * scale)
            
        env_shape = np.ones(n_samples, dtype=np.float32)
        if att_s > 0: env_shape[:att_s] = np.sin(np.linspace(0, np.pi/2, att_s))
        if rel_s > 0: env_shape[-rel_s:] = np.cos(np.linspace(0, np.pi/2, rel_s))

        if chunk.ndim > 1: env_shape = env_shape[:, None]
        chunk = chunk * env_shape * 0.7
        
        # Reverb
        chunk = MicroEngine.apply_reverb(chunk, sr, params.get('verb', 0.0) * 0.4)

        # Clicks (Rhythmic)
        if params.get('clicks', False):
            # Click generation logic (simplified/adapted for stereo safety)
            c_len = min(len(chunk), 2000) 
            if c_len > 100:
                raw_src = chunk if chunk.ndim == 1 else chunk[:, 0]
                raw = raw_src[100:c_len][::2]
                click_sig = np.diff(raw, prepend=0) 
                max_val = np.max(np.abs(click_sig))
                if max_val > 1e-5: click_sig = click_sig / max_val
                
                c_env = np.exp(-np.linspace(0, 15, len(click_sig)))
                click_sig = click_sig * c_env * 2.5 
                
                if chunk.ndim > 1:
                    click_sig = np.column_stack((click_sig, click_sig))

                grid_step = int(sr * 0.25)
                for pos in range(0, len(chunk), grid_step):
                    if random.random() < 0.35:
                        remaining = len(chunk) - pos
                        write_len = min(len(click_sig), remaining)
                        if write_len > 0:
                            segment = chunk[pos:pos+write_len] + click_sig[:write_len]
                            chunk[pos:pos+write_len] = np.clip(segment, -1.0, 1.0)

        # [Stereo Conversion & Pan]
        if chunk.ndim == 1:
            chunk_stereo = np.column_stack((chunk, chunk))
        else:
            chunk_stereo = chunk

        pan_depth = params.get('pan', 0.0)
        if pan_depth > 0.01:
            t = np.linspace(0, len(chunk)/sr, len(chunk))
            lfo = np.sin(2 * np.pi * 1.5 * t)
            width = pan_depth * 0.6
            
            l_gain = 1.0 - (width * (0.5 + 0.5 * lfo))
            r_gain = 1.0 - (width * (0.5 + 0.5 * -lfo))
            
            chunk_stereo[:, 0] *= l_gain
            chunk_stereo[:, 1] *= r_gain

        # --- GLOBAL SOFT LIMITER ---
        
        # 1. Soft Saturation
        # np.tanh (Hyperbolic Tangent) is linear at low volumes but curves 
        # smoothly as it approaches +/- 1.0. This behaves like analog tape saturation:
        # it prevents digital clipping by rounding off peaks instead of chopping them.
        chunk_stereo = np.tanh(chunk_stereo)

        # 2. Hard Safety Clamp
        # Explicitly clip to 0.99 to leave a tiny bit of headroom for the DAC (Digital to Analog Converter)
        # to prevent inter-sample peaks.
        chunk_stereo = np.clip(chunk_stereo, -0.99, 0.99)
        
        return chunk_stereo

class ExportThread(QThread):
    finished_ok = pyqtSignal(object, int)
    def __init__(self, data, sr, region, params):
        super().__init__()
        self.data, self.sr, self.region, self.params = data, sr, region, params
    def run(self):
        try:
            processed = MicroEngine.process_grain(self.data, self.sr, self.region, self.params)
            self.finished_ok.emit(processed, self.sr)
        except: pass

# --- UI COMPONENTS ---

STYLES = """
    QMainWindow { background-color: #f6f9fc; }
    QLabel { color: #64748b; font-family: 'Segoe UI', sans-serif; font-size: 11px; font-weight: 600; }
    QLabel#ValLabel { color: #3f6c9b; }
"""

class HeaderCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.phase = 0.0
        self.speed = 0.05
        self.dots = []
        
        # Sparse Dots (Vignette effect: fewer dots near corners)
        cw, ch = 600, 34 
        cx, cy = cw/2, ch/2
        max_dist = math.sqrt(cx*cx + cy*cy)
        
        for i in range(250):
            x = random.randint(0, cw)
            y = random.randint(0, ch)
            
            # Distance factor 0.0 (center) to 1.0 (corner)
            dist = math.sqrt((x-cx)**2 + (y-cy)**2) / max_dist
            prob = 1.0 - (dist * 0.8) # 20% chance at corners, 100% at center
            
            if random.random() < prob:
                self.dots.append([x, y, random.randint(1, 2), random.random(), random.random()*6.28])

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)

    def animate(self):
        self.phase += self.speed
        if self.speed > 0.05:
            self.speed *= 0.95 # Decay speed back to normal
        self.update()

    def intensify(self):
        self.speed = 0.4 # Jump speed

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # 1. Dots
        for d in self.dots:
            x, y, sz, hue, off = d
            # Calculate alpha and clamp it between 0.0 and 1.0
            raw_alpha = 0.2 + 0.3 * math.sin(self.phase + off)
            alpha = max(0.0, min(1.0, raw_alpha))
            
            c = QColor.fromHslF(hue, 0.6, 0.7, alpha)
            painter.setBrush(c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, y), sz, sz)

        # 2. Smooth Sine Wave with Fade extremities
        path = QPainterPath()
        amp = 6
        freq = 0.02
        cy = h / 2
        path.moveTo(0, cy)
        
        for i in range(0, w, 2):
            y = cy + math.sin(i * freq + self.phase) * amp
            path.lineTo(i, y)
            
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(0, 0, 0, 0))
        grad.setColorAt(0.2, QColor.fromHslF((self.phase*0.1)%1.0, 0.6, 0.75, 1.0))
        grad.setColorAt(0.8, QColor.fromHslF((self.phase*0.1+0.5)%1.0, 0.6, 0.75, 1.0))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        
        pen = QPen(QBrush(grad), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

class PastelToggle(QWidget):
    stateChanged = pyqtSignal(bool)
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.checked = False
        self.label = label_text
        self.phase = random.random()
        self.anim_t = QTimer(self)
        self.anim_t.timeout.connect(self.update)
        self.anim_t.start(50)

    def mousePressEvent(self, e):
        self.checked = not self.checked
        self.stateChanged.emit(self.checked)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        
        self.phase = (self.phase + 0.02) % 1.0
        
        # Background
        bg_col = QColor("#e2e8f0")
        if self.checked:
            # Gradient
            h1 = self.phase
            h2 = (self.phase + 0.3) % 1.0
            grad = QLinearGradient(0,0,r.width(),0)
            grad.setColorAt(0, QColor.fromHslF(h1, 0.6, 0.85))
            grad.setColorAt(1, QColor.fromHslF(h2, 0.6, 0.85))
            brush = QBrush(grad)
        else:
            brush = QBrush(bg_col)
            
        p.setBrush(brush)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 12, 12)
        
        # Circle Indicator
        cx = 12 if not self.checked else r.width() - 12
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(cx, r.height()/2), 8, 8)
        
        # Text - Modified for subtler opacity
        txt_col = QColor("#64748b") 
        txt_col.setAlpha(180) # Reduced opacity for subtle look
        if self.checked: 
            txt_col = QColor("#3f6c9b")
            txt_col.setAlpha(230)

        p.setPen(txt_col)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        p.setFont(font)
        
        # Centered text (approximate offset based on state)
        if self.checked:
            p.drawText(QRectF(0, 0, r.width()-24, r.height()), Qt.AlignmentFlag.AlignCenter, self.label)
        else:
            p.drawText(QRectF(24, 0, r.width()-24, r.height()), Qt.AlignmentFlag.AlignCenter, self.label)

class PastelPush(QWidget):
    clicked = pyqtSignal()
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.label = label_text
        self.pressed = False

    def mousePressEvent(self, e):
        self.pressed = True
        self.update()

    def mouseReleaseEvent(self, e):
        if self.pressed:
            self.clicked.emit()
            self.pressed = False
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        
        # Background - slightly darker when pressed
        bg_col = QColor("#cbd5e1") if self.pressed else QColor("#e2e8f0")
        p.setBrush(QBrush(bg_col))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 12, 12)
        
        # Text - Subtle like the toggles
        txt_col = QColor("#64748b")
        txt_col.setAlpha(180) 
        p.setPen(txt_col)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        p.setFont(font)
        
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.label)

class PastelExportButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(30) # Subtler height
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)
        self.hovered = False

    def animate(self):
        self.phase = (self.phase + 0.01) % 1.0
        if self.hovered: self.update()

    def enterEvent(self, e): self.hovered = True; self.update()
    def leaveEvent(self, e): self.hovered = False; self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        grad = QLinearGradient(0, 0, rect.width(), 0)
        for i in range(3):
            t = i / 2.0
            h = (self.phase + t * 0.2) % 1.0
            s = 0.6 if self.hovered else 0.4
            l = 0.85 if self.hovered else 0.94
            grad.setColorAt(t, QColor.fromHslF(h, s, l))
        
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 4, 4)
        
        painter.setPen(QColor("#94a3b8")) # Subtler text color
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold)) # Subtler font
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())

class PrismSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFixedHeight(20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.phase = random.random()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            val = self.minimum() + ((self.maximum()-self.minimum()) * e.pos().x()) / self.width()
            self.setValue(int(val))
            e.accept()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            val = self.minimum() + ((self.maximum()-self.minimum()) * e.pos().x()) / self.width()
            self.setValue(int(val))
            e.accept()
        super().mouseMoveEvent(e)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        track_y = int(rect.center().y())
        start_x = int(rect.left() + 8)
        end_x = int(rect.right() - 8)
        w = end_x - start_x
        
        painter.setPen(QPen(QColor("#f1f5f9"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start_x, track_y, end_x, track_y)
        
        val_norm = (self.value() - self.minimum()) / (self.maximum() - self.minimum() + 1e-9)
        fill_x = int(start_x + val_norm * w)
        
        grad = QLinearGradient(start_x, 0, end_x, 0)
        h1 = (self.phase) % 1.0
        h2 = (self.phase + 0.3) % 1.0
        grad.setColorAt(0, QColor.fromHslF(h1, 0.6, 0.8))
        grad.setColorAt(1, QColor.fromHslF(h2, 0.6, 0.8))
        
        pen = QPen(QBrush(grad), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(start_x, track_y, fill_x, track_y)
        
        # Handle
        painter.setBrush(QColor("white"))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawEllipse(QPointF(float(fill_x), float(track_y)), 6, 6)
        
        painter.setBrush(QColor("#64748b"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(float(fill_x), float(track_y)), 2, 2)

class ZoomWaveEditor(QWidget):
    selection_changed = pyqtSignal(tuple)
    import_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.data = None
        self.sel_start, self.sel_end = 0.3, 0.7
        self.play_head = -1
        self.zoom_level, self.view_offset = 1.0, 0.0
        self.is_dragging, self.drag_mode = False, None
        self.last_mouse_x = 0
        self.hue_anim = 0.0
        # Optimization: Cache container
        self.cached_px = None

    def advance_anim(self):
        self.hue_anim = (self.hue_anim + 0.005) % 1.0
        self.update()

    def resizeEvent(self, e):
        self.cached_px = None # Invalidate cache on resize
        super().resizeEvent(e)

    def set_data(self, data):
        self.data = data
        self.zoom_level, self.view_offset = 1.0, 0.0
        self.cached_px = None # Invalidate cache
        self.update()

    def set_play_head(self, pos):
        self.play_head = pos
        if pos >= 0: self.advance_anim()
        else: self.update()

    def wheelEvent(self, e):
        if self.data is None: return
        mx, w = e.position().x(), self.width()
        vw = 1.0 / self.zoom_level
        mn = self.view_offset + (mx/w)*vw
        zf = 1.25 if e.angleDelta().y() > 0 else 0.75
        
        self.zoom_level = max(1.0, min(50.0, self.zoom_level * zf))
        nvw = 1.0 / self.zoom_level
        self.view_offset = max(0, min(1.0 - nvw, mn - (mx/w)*nvw))
        
        self.cached_px = None # Invalidate cache
        self.update()

    def mousePressEvent(self, e):
        if self.data is None: return self.import_requested.emit()
        
        self.last_mouse_x = e.pos().x()
        w = self.width()
        vw = 1.0 / self.zoom_level
        
        # Calculate current Selection in pixels
        s, end = min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)
        s_px = (s - self.view_offset) / vw * w
        e_px = (end - self.view_offset) / vw * w
        
        # Hit detection tolerance
        tol = 10 
        
        if e.button() == Qt.MouseButton.RightButton:
            self.drag_mode = 'pan'
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            # 1. Check Edges first (Resize)
            if abs(e.pos().x() - s_px) < tol:
                self.drag_mode = 'resize_l'
            elif abs(e.pos().x() - e_px) < tol:
                self.drag_mode = 'resize_r'
            # 2. Check Inside (Move)
            elif s_px < e.pos().x() < e_px:
                self.drag_mode = 'move'
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            # 3. Outside (New Selection)
            else:
                self.drag_mode = 'new'
                cn = self.view_offset + (e.pos().x()/w)*vw
                self.sel_start = self.sel_end = max(0.0, min(1.0, cn))
            
            self.is_dragging = True
            self.update()

    def mouseMoveEvent(self, e):
        if self.data is None: return
        x, w = e.pos().x(), self.width()
        vw = 1.0 / self.zoom_level
        
        # --- Hover Cursor Logic (When not dragging) ---
        if not self.is_dragging:
            s, end = min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)
            s_px = (s - self.view_offset) / vw * w
            e_px = (end - self.view_offset) / vw * w
            tol = 10
            
            if abs(x - s_px) < tol or abs(x - e_px) < tol:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif s_px < x < e_px:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            return

        # --- Dragging Logic ---
        dx_px = x - self.last_mouse_x
        dx_norm = (dx_px / w) * vw
        
        if self.drag_mode == 'pan':
            self.view_offset = max(0, min(1.0 - vw, self.view_offset - dx_norm))
            self.cached_px = None
            
        elif self.drag_mode == 'move':
            # Calculate width to ensure we don't collapse it
            width = abs(self.sel_end - self.sel_start)
            # Apply delta
            new_s = min(self.sel_start, self.sel_end) + dx_norm
            new_e = new_s + width
            
            # Boundary checks
            if new_s < 0.0:
                new_s = 0.0
                new_e = width
            elif new_e > 1.0:
                new_e = 1.0
                new_s = 1.0 - width
                
            self.sel_start, self.sel_end = new_s, new_e
            self.selection_changed.emit((self.sel_start, self.sel_end))
            
        elif self.drag_mode == 'resize_l':
            # Adjust start, allow crossing over
            # We assume sel_start is the 'left' visual handle for simplicity in storage,
            # but we use min/max in logic. Here we just update the specific handle found in Press.
            # To simplify: we assume the user intends to move the boundary closest to them.
            cur_min = min(self.sel_start, self.sel_end)
            cur_max = max(self.sel_start, self.sel_end)
            new_min = max(0.0, min(cur_max, cur_min + dx_norm)) # Clamp to 0 and other edge
            
            self.sel_start, self.sel_end = new_min, cur_max
            self.selection_changed.emit((self.sel_start, self.sel_end))
            
        elif self.drag_mode == 'resize_r':
            cur_min = min(self.sel_start, self.sel_end)
            cur_max = max(self.sel_start, self.sel_end)
            new_max = max(cur_min, min(1.0, cur_max + dx_norm)) # Clamp to other edge and 1
            
            self.sel_start, self.sel_end = cur_min, new_max
            self.selection_changed.emit((self.sel_start, self.sel_end))
            
        elif self.drag_mode == 'new':
            cn = self.view_offset + (x/w)*vw
            self.sel_end = max(0.0, min(1.0, cn))
            self.selection_changed.emit((min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)))

        self.last_mouse_x = x
        self.update()
            
    def mouseReleaseEvent(self, e):
        # 1. Capture state before resetting
        was_active = self.is_dragging
        
        # 2. Reset Dragging Flags immediately
        # This is crucial: MainWindow checks this flag to decide whether to play audio.
        self.is_dragging = False
        self.drag_mode = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        if self.data is not None:
            # 3. Normalize Selection (ensure start < end)
            if self.sel_start > self.sel_end:
                self.sel_start, self.sel_end = self.sel_end, self.sel_start
            
            # 4. Trigger Playback
            # We emit the signal now. Since self.is_dragging is False, 
            # the MainWindow's auto_prev() will accept this and start the timer.
            if was_active:
                self.selection_changed.emit((self.sel_start, self.sel_end))
        else:
            # 5. Handle empty state click (Import)
            self.import_requested.emit()

    def update_cache(self):
        # Draw the heavy waveform into a Pixmap once
        dpr = self.devicePixelRatio()
        self.cached_px = QPixmap(self.size() * dpr)
        self.cached_px.setDevicePixelRatio(dpr)
        self.cached_px.fill(QColor("#f6f9fc"))
        
        painter = QPainter(self.cached_px)
        rect = self.rect()
        w, h = self.width(), self.height()

        if self.data is None:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "drag sample or click to load")
            painter.end()
            return

        vw = 1.0 / self.zoom_level
        start_idx = int(self.view_offset * len(self.data))
        end_idx = int((self.view_offset + vw) * len(self.data))
        
        # Clamp
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data), end_idx)
        
        vdata = self.data[start_idx:end_idx]
        if len(vdata) == 0: 
            painter.end()
            return

        # Path Gen
        step = max(1, len(vdata) // w) 
        path = QPainterPath()
        cy = h / 2
        path.moveTo(0, cy)
        amp = h * 0.45
        
        # Using numpy for speed in loop
        x_steps = np.linspace(0, w, len(vdata[::step]))
        y_steps = cy - vdata[::step] * amp
        
        path.moveTo(0, cy)
        for i in range(len(x_steps)):
            path.lineTo(x_steps[i], y_steps[i])

        grad_wave = QLinearGradient(0, 0, w, 0)
        grad_wave.setColorAt(0.0, QColor("#ff9a9e"))
        grad_wave.setColorAt(0.5, QColor("#fad0c4"))
        grad_wave.setColorAt(1.0, QColor("#a18cd1"))
        
        wave_pen = QPen(QBrush(grad_wave), 1.5)
        wave_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(wave_pen)
        painter.drawPath(path)
        painter.end()

    def paintEvent(self, event):
        # 1. Rebuild Cache if needed
        if self.cached_px is None:
            self.update_cache()

        painter = QPainter(self)
        w, h = self.width(), self.height()

        # 2. Draw Cached Background (Zero cost)
        painter.drawPixmap(0, 0, self.cached_px)

        if self.data is None: return

        # 3. Draw Selection & Playhead (Dynamic elements)
        vw = 1.0 / self.zoom_level
        s = min(self.sel_start, self.sel_end)
        e = max(self.sel_start, self.sel_end)
        
        sx = (s - self.view_offset) / vw * w
        ex = (e - self.view_offset) / vw * w
        sel_w = ex - sx

        if sel_w > 1:
            # Light selection rectangle overlay
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(63, 108, 155, 10))
            painter.drawRect(QRectF(sx, 0, sel_w, h))

            # Handles
            grad_h = QLinearGradient(0, 0, 0, h)
            h1 = self.hue_anim
            h2 = (self.hue_anim + 0.3) % 1.0
            grad_h.setColorAt(0, QColor.fromHslF(h1, 0.5, 0.8))
            grad_h.setColorAt(1, QColor.fromHslF(h2, 0.5, 0.8))
            brush = QBrush(grad_h)
            
            painter.fillRect(QRectF(sx, 0, 3, h), brush)
            painter.fillRect(QRectF(ex-3, 0, 3, h), brush)

        if self.play_head >= 0:
            ph_abs = s + self.play_head * (e - s)
            px = (ph_abs - self.view_offset) / vw * w
            
            grad_ph = QLinearGradient(0, 0, 0, h)
            grad_ph.setColorAt(0.0, QColor(160, 190, 255, 200))
            grad_ph.setColorAt(1.0, QColor(255, 180, 200, 200))
            
            painter.setBrush(QBrush(grad_ph))
            painter.drawRect(QRectF(px - 1.25, 0, 2.5, h))

class ControlRow(QWidget):
    valueChanged = pyqtSignal(float)
    def __init__(self, label, min_v, max_v, default_v, fmt="{:.2f}"):
        super().__init__()
        self.min_v, self.max_v = min_v, max_v
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel(label.lower())
        lbl.setFixedWidth(45)
        
        self.val = QLabel(fmt.format(default_v))
        self.val.setObjectName("ValLabel")
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.val.setFixedWidth(35)
        
        self.slider = PrismSlider()
        self.slider.setRange(0, 1000)
        self.slider.setValue(int((default_v - min_v)/(max_v - min_v)*1000))
        self.slider.valueChanged.connect(lambda v: self.update_val(v, fmt))
        
        layout.addWidget(lbl)
        layout.addWidget(self.slider)
        layout.addWidget(self.val)
        
    def update_val(self, v, fmt):
        real = self.min_v + (v/1000)*(self.max_v - self.min_v)
        self.val.setText(fmt.format(real))
        self.valueChanged.emit(real)

    def value(self):
        return self.min_v + (self.slider.value()/1000)*(self.max_v - self.min_v)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("micro")
        self.resize(650, 520)
        self.setAcceptDrops(True)
        self.data, self.sr, self.p_data, self.tf = None, 44100, None, None
        self.c_reg = (0.3, 0.7)
        
        self.player = QMediaPlayer()
        self.last_media_pos = 0
        self.last_update_time = 0
        self.ao = QAudioOutput()
        self.player.setAudioOutput(self.ao)
        self.player.mediaStatusChanged.connect(self.media_status)
        
        cw = QWidget()
        self.setCentralWidget(cw)
        mv = QVBoxLayout(cw)
        mv.setContentsMargins(25, 15, 25, 20)
        mv.setSpacing(12)
        
        # 1. Header
        self.header = HeaderCanvas()
        mv.addWidget(self.header)
        
        # 2. Waveform
        wfr = QFrame()
        wfr.setStyleSheet("background: #f6f9fc; border-radius: 8px; border: 1px solid #e2e8f0;")
        wl = QVBoxLayout(wfr); wl.setContentsMargins(1,1,1,1)
        self.wave = ZoomWaveEditor()
        self.wave.import_requested.connect(self.open_file)
        self.wave.selection_changed.connect(self.set_reg)
        wl.addWidget(self.wave)
        mv.addWidget(wfr, 1)
        
        # 3. Controls 
        crow = QHBoxLayout(); crow.setSpacing(30)
        
        c1 = QVBoxLayout(); c1.setSpacing(4)
        self.k_att = ControlRow("attack", 0, 0.5, 0.01)
        self.k_rel = ControlRow("release", 0, 0.5, 0.1)
        self.k_pan = ControlRow("pan", 0.0, 1.0, 0.0) 
        self.k_verb = ControlRow("reverb", 0, 1, 0.0)
        for w in [self.k_att, self.k_rel, self.k_pan, self.k_verb]: c1.addWidget(w)
        c1.addStretch()
        
        c2 = QVBoxLayout(); c2.setSpacing(4)
        self.k_pt = ControlRow("rate", 0.5, 2.0, 1.0, "{:.2f}x")
        self.k_tn = ControlRow("tone", 0, 1, 1.0)
        self.k_cr = ControlRow("crush", 0, 1, 0.0)
        self.k_cmp = ControlRow("comp", 0, 1, 0.0) 
        for w in [self.k_pt, self.k_tn, self.k_cr, self.k_cmp]: c2.addWidget(w)
        c2.addStretch()
        
        crow.addLayout(c1, 1); crow.addLayout(c2, 1)
        mv.addLayout(crow)
        
        # 4. Reverse, Repeat, Click & Clear
        t_row = QHBoxLayout()
        t_row.addStretch()
        
        # Clear Button
        self.btn_clear = PastelPush("clear")
        self.btn_clear.clicked.connect(self.clear_state)
        t_row.addWidget(self.btn_clear)
        
        t_row.addSpacing(10)

        # Click Toggle (New)
        self.t_clk = PastelToggle("click")
        self.t_clk.stateChanged.connect(self.auto_prev)
        t_row.addWidget(self.t_clk)

        t_row.addSpacing(10)

        # Reverse Toggle
        self.t_rev = PastelToggle("reverse")
        t_row.addWidget(self.t_rev)
        
        t_row.addSpacing(10)

        # Repeat Toggle
        self.t_rep = PastelToggle("repeat")
        self.t_rep.stateChanged.connect(self.auto_prev)
        t_row.addWidget(self.t_rep)
        
        t_row.addStretch()
        mv.addLayout(t_row)
        
        # Bottom Export
        self.btn_ex = PastelExportButton("export")
        self.btn_ex.clicked.connect(self.export)
        mv.addWidget(self.btn_ex)
        # Signals
        for k in [self.k_att, self.k_rel, self.k_pan, self.k_verb, self.k_pt, self.k_tn, self.k_cr, self.k_cmp]:
            k.valueChanged.connect(self.auto_prev)
        self.t_rev.stateChanged.connect(self.auto_prev)

        self.ptimer = QTimer(interval=150, singleShot=True, timeout=self.prev)
        self.atimer = QTimer(interval=16, timeout=self.tick)

    def set_reg(self, r): self.c_reg = r; self.auto_prev()
    def auto_prev(self): 
        if not self.wave.is_dragging: self.ptimer.start()
        
    def get_p(self):
        return {'attack': self.k_att.value(), 'release': self.k_rel.value(),
                'pan': self.k_pan.value(),
                'reverse': self.t_rev.checked,
                'rate': self.k_pt.value(), 'tone': self.k_tn.value(),
                'crush': self.k_cr.value(), 
                'compress': self.k_cmp.value(),
                'verb': self.k_verb.value(), 
                'clicks': self.t_clk.checked}

    def prev(self):
        if self.data is None: return
        self.player.stop()
        self.th = ExportThread(self.data, self.sr, self.c_reg, self.get_p())
        self.th.finished_ok.connect(self.on_fin)
        self.th.start()

    def on_fin(self, d, sr):
        self.p_data = d 
        try:
            fd, p = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            
            save_data = d
            if self.t_rep.checked and len(d) > 0:
                target_len = sr * 30
                repeats = max(2, int(target_len / len(d)))
                repeats = min(repeats, 64)
                
                # Check for Stereo (2D) vs Mono (1D) for correct tiling
                if d.ndim > 1:
                    save_data = np.tile(d, (repeats, 1))
                else:
                    save_data = np.tile(d, repeats)

            MicroEngine.save_file(p, save_data, sr)
            self.tf = p
            
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(p))
            self.player.play()
            self.atimer.start()
        except: pass

    def tick(self):
        # 1. Always calculate position if we have data, regardless of play state
        if self.p_data is None or len(self.p_data) == 0:
            return

        state = self.player.playbackState()
        
        # Logic: If playing, update form hardware. If stopped, keep at 0 or last known?
        # Let's simply loop the visual if playing, reset to start if stopped.
        
        if state == QMediaPlayer.PlaybackState.PlayingState:
            raw_pos = self.player.position()
            now = time.time()

            if raw_pos != self.last_media_pos:
                self.last_media_pos = raw_pos
                self.last_update_time = now
                est_pos = raw_pos
            else:
                elapsed_ms = (now - self.last_update_time) * 1000.0
                est_pos = self.last_media_pos + elapsed_ms
            
            # Map loop time to visual selection
            # The audio generated is a Loop. 
            # We map the 0-100% of the loop playback to the start-end of the selection.
            single_dur_ms = (len(self.p_data) / self.sr) * 1000.0
            if single_dur_ms > 0:
                norm_pos = (est_pos % single_dur_ms) / single_dur_ms
                self.wave.set_play_head(norm_pos)
        else:
            # If stopped, just ensure playhead is visible at start of selection
            self.wave.set_play_head(0.0)

    def media_status(self, s):
        if s == QMediaPlayer.MediaStatus.EndOfMedia:
            if self.t_rep.checked:
                self.player.setPosition(0)
                self.player.play()
            else:
                self.wave.set_play_head(-1); self.atimer.stop()

    def clear_state(self):
        self.player.stop()
        self.wave.set_play_head(-1)
        self.atimer.stop()
        self.data, self.p_data = None, None
        self.wave.set_data(None)
        # Reset file reference to ensure clean slate
        self.player.setSource(QUrl())

    def export(self):
        if self.p_data is None: return
        n = f"micro_export_{int(time.time())}.wav"
        try: 
            MicroEngine.save_file(os.path.join(os.getcwd(), n), self.p_data, self.sr)
            self.header.intensify() # Trigger visual feedback
            self.btn_ex.setText(f"saved {n}")
            QTimer.singleShot(2000, lambda: self.btn_ex.setText("export wav"))
        except: pass

    def dragEnterEvent(self, e): e.accept() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): self.load_p(e.mimeData().urls()[0].toLocalFile())
    def open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "open", "", "audio (*.wav *.mp3 *.flac *.ogg)")
        if p: self.load_p(p)
    def load_p(self, p):
        try:
            d, sr = MicroEngine.load_file(p); self.data, self.sr = d, sr
            self.wave.set_data(d); self.prev()
        except: pass
    def closeEvent(self, e):
        if self.tf and os.path.exists(self.tf): 
            try: os.remove(self.tf)
            except: pass
        e.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLES)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())