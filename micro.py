import ctypes
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
                         QBrush, QFont, QPixmap, QCursor, QPolygonF, QRadialGradient, QIcon)
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
        tail_len = int(sr * 1.5) 
        noise = np.random.randn(tail_len)
        env = np.exp(-np.linspace(0, 1, tail_len) * 6.0)
        ir = noise * env
        b, a = signal.butter(1, 0.4, 'low')
        ir = signal.lfilter(b, a, ir)
        
        padding = np.zeros((tail_len, x.shape[1] if x.ndim > 1 else 1), dtype=np.float32)
        if x.ndim == 1: x = x[:, None]
        padded_x = np.concatenate([x, padding])
        
        wet = signal.fftconvolve(padded_x, ir[:, None] if ir.ndim==1 else ir, mode='full')
        
        wet = wet[:len(padded_x)]
        wet = wet / (np.max(np.abs(wet)) + 1e-9)
        
        out = (1 - mix) * padded_x + mix * wet
        return out

    @staticmethod
    def get_zero_crossing(data, target_idx, search_window=1024):
        start = max(0, target_idx - search_window // 2)
        end = min(len(data), target_idx + search_window // 2)
        if start >= end: return target_idx
        chunk = data[start:end]
        if chunk.ndim > 1: amp_profile = np.sum(np.abs(chunk), axis=1)
        else: amp_profile = np.abs(chunk)
        min_local_idx = np.argmin(amp_profile)
        return start + min_local_idx

    @staticmethod
    def precompute_heavy_fx(data, sr, params):
        chunk = data.copy()
        
        crush = params.get('crush', 0.0)
        if crush > 0.01:
            factor = 1.0 - (crush * 0.98) 
            target_len = int(len(chunk) * factor)
            if target_len > 5:
                down = signal.resample(chunk, target_len)
                chunk = signal.resample(down, len(chunk))

        tone = params.get('tone', 0.5)
        if abs(tone - 0.5) > 0.05:
            if tone < 0.5:
                norm = tone * 2.0
                cutoff = 100 * (180 ** norm)
                sos = signal.butter(2, cutoff, 'low', fs=sr, output='sos')
                chunk = signal.sosfilt(sos, chunk)
            else:
                norm = (tone - 0.5) * 2.0
                cutoff = 20 + (8000 * (norm ** 2))
                sos = signal.butter(2, cutoff, 'high', fs=sr, output='sos')
                chunk = signal.sosfilt(sos, chunk)

        comp = params.get('compress', 0.0)
        if comp > 0.01:
            sos_env = signal.butter(1, 0.02, output='sos')
            det_sig = np.abs(chunk)
            env = signal.sosfilt(sos_env, det_sig)
            thresh = 0.6 * (1.0 - comp * 0.7)
            gain_red = np.minimum(1.0, thresh / (env + 1e-6))
            chunk = chunk * gain_red * (1.0 + comp * 2.5)

        return chunk

    @staticmethod
    def render_playback(source_data, sr, region, params, abort_check=None): # <--- Add arg
        is_repeat = params.get('repeat', False)
        grain_map = []
        
        def process_slice(sub_region):
            raw_start = int(sub_region[0] * len(source_data))
            raw_end = int(sub_region[1] * len(source_data))
            
            s_idx = MicroEngine.get_zero_crossing(source_data, raw_start, 1024)
            e_idx = MicroEngine.get_zero_crossing(source_data, raw_end, 1024)
            
            if s_idx >= e_idx: return np.zeros((1024, 2), dtype=np.float32), s_idx, s_idx
            
            chunk = source_data[s_idx:e_idx].copy()
            
            fade_len = min(int(sr * 0.05), len(chunk) // 2)
            if fade_len > 0:
                fade = np.linspace(0, 1, fade_len)
                chunk[:fade_len] *= fade
                chunk[-fade_len:] *= fade[::-1]

            if params.get('reverse', False): chunk = chunk[::-1]

            rate = params.get('rate', 1.0)
            if abs(rate - 1.0) > 0.01:
                new_len = int(len(chunk) / rate)
                if new_len > 0: chunk = signal.resample(chunk, new_len)

            att_p = params.get('attack', 0.01)
            rel_p = params.get('release', 0.01)
            n_s = len(chunk)
            att_s = int(n_s * att_p)
            rel_s = int(n_s * rel_p)
            if att_s + rel_s > n_s:
                scale = n_s / (att_s + rel_s + 1)
                att_s, rel_s = int(att_s * scale), int(rel_s * scale)
            
            env = np.ones(n_s, dtype=np.float32)
            if att_s > 0: env[:att_s] = np.sin(np.linspace(0, np.pi/2, att_s))
            if rel_s > 0: env[-rel_s:] = np.cos(np.linspace(0, np.pi/2, rel_s))
            chunk *= env * 0.8
            
            if params.get('clicks', False):
                grid = int(sr * 0.15)
                for pos in range(0, len(chunk), grid):
                    if random.random() < 0.3:
                        click_len = random.randint(400, 1200) 
                        if pos + click_len < len(chunk):
                            freq = random.uniform(300, 1200)
                            t = np.arange(click_len) / sr
                            tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
                            c_env = np.exp(-np.linspace(0, 6, click_len))
                            burst = tone * c_env * 0.25
                            chunk[pos:pos+click_len] += burst

            if chunk.ndim == 1: chunk = np.column_stack((chunk, chunk))
            
            pan = params.get('pan', 0.0)
            if pan > 0.01:
                rng = random.uniform(-1, 1)
                w = pan * 0.8
                l = 1.0 - (w * (0.5 + 0.5 * rng))
                r = 1.0 - (w * (0.5 + 0.5 * -rng))
                chunk[:, 0] *= l
                chunk[:, 1] *= r

            return np.tanh(chunk), s_idx, e_idx

        final_buffer = None
        
        if not is_repeat:
            if abort_check and abort_check(): return None, [] # <--- Check here
            chunk, s, e = process_slice(region)
            final_buffer = chunk
            grain_map.append((0, (len(chunk)/sr)*1000.0, region[0], region[1]))
        else:
            bpm = random.randint(90, 120)
            beat_sec = 60.0 / bpm
            total_slots = 32
            seq_len = int(total_slots * (beat_sec/4.0) * sr)
            seq_buffer = np.zeros((seq_len, 2), dtype=np.float32)
            
            count = random.randint(12, 24)
            slots = sorted([random.randint(0, total_slots-1) for _ in range(count)])
            reg_len = region[1] - region[0]
            
            for slot in slots:
                if abort_check and abort_check(): return None, [] # <--- Check inside loop
                offset = random.uniform(0, max(0, reg_len - 0.001))
                s_pt = region[0] + offset
                dur = random.uniform(2.0, 8.0) * (beat_sec/4.0)
                len_norm = dur * sr / len(source_data)
                e_pt = min(region[1], s_pt + len_norm)
                
                grain, s_idx, e_idx = process_slice((s_pt, e_pt))
                if len(grain) == 0: continue
                
                ins_idx = int(slot * (beat_sec/4.0) * sr)
                w_len = min(len(grain), seq_len - ins_idx)
                if w_len > 0:
                    seq_buffer[ins_idx:ins_idx+w_len] += grain[:w_len]
                    
                    p_s = (ins_idx/sr)*1000.0
                    p_e = ((ins_idx+w_len)/sr)*1000.0
                    n_s = s_idx/len(source_data)
                    n_e = (s_idx+w_len)/len(source_data)
                    grain_map.append((p_s, p_e, n_s, n_e))
            
            pk = np.max(np.abs(seq_buffer))
            if pk > 0.9: seq_buffer *= (0.9/pk)
            final_buffer = seq_buffer

        # Reverb is the most expensive operation.
        v_amt = params.get('verb', 0.0)
        if v_amt > 0.01:
            if abort_check and abort_check(): return None, [] # <--- Check before reverb
            final_buffer = MicroEngine.apply_reverb(final_buffer, sr, v_amt * 0.6)
        
        return np.clip(final_buffer, -0.99, 0.99), grain_map

class ExportThread(QThread):
    finished_ok = pyqtSignal(object, int, list, object) 
    
    def __init__(self, raw_data, sr, region, params, cached_source=None, cached_hash=None):
        super().__init__()
        self.raw = raw_data
        self.sr = sr
        self.reg = region
        self.params = params
        self.cached = cached_source
        self.in_hash = cached_hash
        self._is_aborted = False  # Flag to stop processing

    def abort(self):
        """Signals the thread to stop calculation immediately."""
        self._is_aborted = True

    def run(self):
        try:
            if self._is_aborted: return

            # Instead of processing the whole song, we only process the loop + padding.
            
            total_len = len(self.raw)
            r_start, r_end = self.reg # Current region (0.0 to 1.0)
            
            # Add 250ms padding to let filters/reverb settle naturally
            pad_samples = int(self.sr * 0.25)
            
            # Calculate integer indices for the slice
            idx_start = max(0, int(r_start * total_len) - pad_samples)
            idx_end = min(total_len, int(r_end * total_len) + pad_samples)
            
            # Create the small work buffer
            sliced_raw = self.raw[idx_start:idx_end]
            if len(sliced_raw) == 0: return

            # We must translate the User's Region (0-1 relative to Song)
            # into a Relative Region (0-1 relative to this specific Slice)
            # so the engine knows where the "active" part of the slice is.
            
            # Start index of selection relative to the slice start
            loc_s = int(r_start * total_len) - idx_start
            loc_e = int(r_end * total_len) - idx_start
            
            # Normalize to the slice length
            rel_reg = (loc_s / len(sliced_raw), loc_e / len(sliced_raw))

            dsp_source = self.cached
            
            if dsp_source is None:
                if self._is_aborted: return 
                # Process only the small slice! (Fast)
                dsp_source = MicroEngine.precompute_heavy_fx(sliced_raw, self.sr, self.params)

            if self._is_aborted: return

            # Render playback using the small slice and relative region
            # (Pass lambda to allow aborting mid-render)
            final, g_map = MicroEngine.render_playback(
                dsp_source, self.sr, rel_reg, self.params, 
                abort_check=lambda: self._is_aborted
            )

            if self._is_aborted or final is None: return

            # The engine returned grain positions relative to the SLICE.
            # We must convert them back to Global coordinates so the UI draws them correctly.
            
            fixed_map = []
            slice_len = len(sliced_raw)
            
            for (ps, pe, ns, ne) in g_map:
                # ns, ne are 0-1 relative to slice
                # Convert to absolute sample index -> normalize to total song length
                abs_s = ns * slice_len + idx_start
                abs_e = ne * slice_len + idx_start
                
                global_ns = abs_s / total_len
                global_ne = abs_e / total_len
                
                fixed_map.append((ps, pe, global_ns, global_ne))

            self.finished_ok.emit(final, self.sr, fixed_map, dsp_source)
            
        except Exception as e:
            if not self._is_aborted:
                print(f"Export Error: {e}")

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
        cw, ch = 600, 34 
        cx, cy = cw/2, ch/2
        max_dist = math.sqrt(cx*cx + cy*cy)
        for i in range(250):
            x = random.randint(0, cw)
            y = random.randint(0, ch)
            dist = math.sqrt((x-cx)**2 + (y-cy)**2) / max_dist
            prob = 1.0 - (dist * 0.8)
            if random.random() < prob:
                self.dots.append([x, y, random.randint(1, 2), random.random(), random.random()*6.28])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)

    def animate(self):
        self.phase += self.speed
        if self.speed > 0.05: self.speed *= 0.95
        self.update()

    def intensify(self): self.speed = 0.4 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        for d in self.dots:
            x, y, sz, hue, off = d
            raw_alpha = 0.2 + 0.3 * math.sin(self.phase + off)
            alpha = max(0.0, min(1.0, raw_alpha))
            c = QColor.fromHslF(hue, 0.6, 0.7, alpha)
            painter.setBrush(c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, y), sz, sz)
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
        bg_col = QColor("#e2e8f0")
        if self.checked:
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
        cx = 12 if not self.checked else r.width() - 12
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(cx, r.height()/2), 8, 8)
        txt_col = QColor("#64748b") 
        txt_col.setAlpha(180) 
        if self.checked: 
            txt_col = QColor("#3f6c9b")
            txt_col.setAlpha(230)
        p.setPen(txt_col)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        p.setFont(font)
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
        bg_col = QColor("#cbd5e1") if self.pressed else QColor("#e2e8f0")
        p.setBrush(QBrush(bg_col))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 12, 12)
        txt_col = QColor("#64748b")
        txt_col.setAlpha(180) 
        p.setPen(txt_col)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.label)

class PastelExportButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(30)
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
        painter.setPen(QColor("#94a3b8"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())
        l_grad = QLinearGradient(0, 0, 15, 0)
        l_grad.setColorAt(0.0, QColor(255, 255, 255, 150))
        l_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(0, 0, 15, rect.height(), l_grad)
        r_grad = QLinearGradient(rect.width()-15, 0, rect.width(), 0)
        r_grad.setColorAt(0.0, QColor(255, 255, 255, 0))
        r_grad.setColorAt(1.0, QColor(255, 255, 255, 150))
        painter.fillRect(rect.width()-15, 0, 15, rect.height(), r_grad)

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
        self.cached_px = None
        
        self.grain_map = [] 
        self.current_loop_ms = -1.0

    def set_grain_map(self, g_map):
        self.grain_map = g_map
        self.update()

    def set_playback_pos(self, ms):
        self.current_loop_ms = ms
        self.update()
    
    def advance_anim(self):
        self.hue_anim = (self.hue_anim + 0.005) % 1.0
        self.update()

    def resizeEvent(self, e):
        self.cached_px = None
        super().resizeEvent(e)

    def set_data(self, data):
        self.data = data
        self.zoom_level, self.view_offset = 1.0, 0.0
        self.cached_px = None
        self.grain_map = []
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
        self.cached_px = None
        self.update()

    def mousePressEvent(self, e):
        if self.data is None: return self.import_requested.emit()
        
        self.last_mouse_x = e.pos().x()
        w = self.width()
        vw = 1.0 / self.zoom_level
        s, end = min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)
        s_px = (s - self.view_offset) / vw * w
        e_px = (end - self.view_offset) / vw * w
        tol = 10 
        
        if e.button() == Qt.MouseButton.RightButton or e.button() == Qt.MouseButton.MiddleButton:
            self.drag_mode = 'pan'
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.is_dragging = True
            return

        if abs(e.pos().x() - s_px) < tol:
            self.drag_mode = 'resize_l'
        elif abs(e.pos().x() - e_px) < tol:
            self.drag_mode = 'resize_r'
        elif s_px < e.pos().x() < e_px:
            self.drag_mode = 'move'
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
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

        dx_px = x - self.last_mouse_x
        dx_norm = (dx_px / w) * vw
        
        if self.drag_mode == 'pan':
            self.view_offset = max(0, min(1.0 - vw, self.view_offset - dx_norm))
            self.cached_px = None
            
        elif self.drag_mode == 'move':
            width = abs(self.sel_end - self.sel_start)
            new_s = min(self.sel_start, self.sel_end) + dx_norm
            new_e = new_s + width
            if new_s < 0.0:
                new_s = 0.0
                new_e = width
            elif new_e > 1.0:
                new_e = 1.0
                new_s = 1.0 - width
            self.sel_start, self.sel_end = new_s, new_e
            self.selection_changed.emit((self.sel_start, self.sel_end))
            
        elif self.drag_mode == 'resize_l':
            cur_min = min(self.sel_start, self.sel_end)
            cur_max = max(self.sel_start, self.sel_end)
            new_min = max(0.0, min(cur_max, cur_min + dx_norm))
            self.sel_start, self.sel_end = new_min, cur_max
            self.selection_changed.emit((self.sel_start, self.sel_end))
            
        elif self.drag_mode == 'resize_r':
            cur_min = min(self.sel_start, self.sel_end)
            cur_max = max(self.sel_start, self.sel_end)
            new_max = max(cur_min, min(1.0, cur_max + dx_norm))
            self.sel_start, self.sel_end = cur_min, new_max
            self.selection_changed.emit((self.sel_start, self.sel_end))
            
        elif self.drag_mode == 'new':
            cn = self.view_offset + (x/w)*vw
            self.sel_end = max(0.0, min(1.0, cn))
            self.selection_changed.emit((min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)))

        self.last_mouse_x = x
        self.update()
            
    def mouseReleaseEvent(self, e):
        was_active = self.is_dragging
        self.is_dragging = False
        self.drag_mode = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        if self.data is not None:
            if self.sel_start > self.sel_end:
                self.sel_start, self.sel_end = self.sel_end, self.sel_start
            if was_active:
                self.selection_changed.emit((self.sel_start, self.sel_end))
        else:
            self.import_requested.emit()

    def update_cache(self):
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
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data), end_idx)
        vdata = self.data[start_idx:end_idx]
        if len(vdata) == 0: 
            painter.end()
            return

        step = max(1, len(vdata) // w) 
        path = QPainterPath()
        cy = h / 2
        path.moveTo(0, cy)
        amp = h * 0.45
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
        if self.cached_px is None: self.update_cache()
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.drawPixmap(0, 0, self.cached_px)

        if self.data is None: return

        vw = 1.0 / self.zoom_level
        s = min(self.sel_start, self.sel_end)
        e = max(self.sel_start, self.sel_end)
        sx = (s - self.view_offset) / vw * w
        ex = (e - self.view_offset) / vw * w
        sel_w = ex - sx

        if sel_w > 1:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(63, 108, 155, 10))
            painter.drawRect(QRectF(sx, 0, sel_w, h))
            grad_h = QLinearGradient(0, 0, 0, h)
            h1 = self.hue_anim
            h2 = (self.hue_anim + 0.3) % 1.0
            grad_h.setColorAt(0, QColor.fromHslF(h1, 0.5, 0.8))
            grad_h.setColorAt(1, QColor.fromHslF(h2, 0.5, 0.8))
            brush = QBrush(grad_h)
            painter.fillRect(QRectF(sx, 0, 3, h), brush)
            painter.fillRect(QRectF(ex-3, 0, 3, h), brush)

        if self.grain_map and self.current_loop_ms >= 0:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            for (p_start, p_end, src_start, src_end) in self.grain_map:
                if p_start <= self.current_loop_ms < p_end:
                    
                    progress = (self.current_loop_ms - p_start) / (p_end - p_start)
                    cur_src = src_start + (src_end - src_start) * progress
                    gx = (cur_src - self.view_offset) / vw * w
                    
                    if 0 <= gx <= w:
                        hue = (cur_src * 5.0 + self.hue_anim) % 1.0
                        col = QColor.fromHslF(hue, 0.6, 0.75, 0.35) 
                        
                        painter.setBrush(col)
                        painter.setPen(QPen(QColor(255, 255, 255, 100), 0.5))
                        
                        tri_h = 55  
                        tri_w = 8   
                        y_base = h  
                        
                        tri = QPolygonF([
                            QPointF(gx, y_base - tri_h),
                            QPointF(gx - tri_w/2, y_base),
                            QPointF(gx + tri_w/2, y_base)
                        ])
                        painter.drawPolygon(tri)

        if self.play_head >= 0 and not self.grain_map:
            ph_abs = s + self.play_head * (e - s)
            px = (ph_abs - self.view_offset) / vw * w
            grad_ph = QLinearGradient(0, 0, 0, h)
            grad_ph.setColorAt(0.0, QColor(160, 190, 255, 200))
            grad_ph.setColorAt(1.0, QColor(255, 180, 200, 200))
            painter.setBrush(QBrush(grad_ph))
            painter.setPen(Qt.PenStyle.NoPen)
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
        self.worker_thread = None      # The currently running thread
        self.update_pending = False    # Do we need to recalc after current finishes?
        self.old_threads = []          # Garbage collection "waiting room"
        self.resize(650, 520)
        self.setAcceptDrops(True)
        self.data, self.sr, self.p_data, self.tf = None, 44100, None, None
        # Thread Locking State
        self.is_processing = False
        self.update_pending = False
        
        self.dsp_cache = None
        self.dsp_hash = None 

        self.dsp_cache = None
        self.dsp_hash = None 
        
        self.grain_map = []
        self.c_reg = (0.3, 0.7)
        self.player = QMediaPlayer()
        
        self.ao = QAudioOutput()
        self.player.setAudioOutput(self.ao)
        self.player.mediaStatusChanged.connect(self.media_status)
        
        # --- New fader animation ---
        self.vol_anim = QPropertyAnimation(self.ao, b"volume")
        self.vol_anim.setDuration(250) # 250ms fade out
        self.vol_anim.setEndValue(0.0)
        self.vol_anim.setEasingCurve(QEasingCurve.Type.Linear)
        # ---------------------------   
       
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
        self.header = HeaderCanvas()
        mv.addWidget(self.header)
        wfr = QFrame()
        wfr.setStyleSheet("background: #f6f9fc; border-radius: 8px; border: 1px solid #e2e8f0;")
        wl = QVBoxLayout(wfr); wl.setContentsMargins(1,1,1,1)
        self.wave = ZoomWaveEditor()
        self.wave.import_requested.connect(self.open_file)
        self.wave.selection_changed.connect(self.set_reg)
        wl.addWidget(self.wave)
        mv.addWidget(wfr, 1)
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
        self.k_tn = ControlRow("tone", 0, 1, 0.5) 
        self.k_cr = ControlRow("crush", 0, 1, 0.0)
        self.k_cmp = ControlRow("comp", 0, 1, 0.0) 
        for w in [self.k_pt, self.k_tn, self.k_cr, self.k_cmp]: c2.addWidget(w)
        c2.addStretch()
        crow.addLayout(c1, 1); crow.addLayout(c2, 1)
        mv.addLayout(crow)
        t_row = QHBoxLayout()
        t_row.addStretch()
        self.btn_clear = PastelPush("clear")
        self.btn_clear.clicked.connect(self.clear_state)
        t_row.addWidget(self.btn_clear)
        t_row.addSpacing(10)
        self.t_clk = PastelToggle("click")
        self.t_clk.stateChanged.connect(self.auto_prev)
        t_row.addWidget(self.t_clk)
        t_row.addSpacing(10)
        self.t_rev = PastelToggle("reverse")
        t_row.addWidget(self.t_rev)
        t_row.addSpacing(10)
        self.t_rep = PastelToggle("grain")
        self.t_rep.stateChanged.connect(self.auto_prev)
        t_row.addWidget(self.t_rep)
        t_row.addStretch()
        mv.addLayout(t_row)
        self.btn_ex = PastelExportButton("export")
        self.btn_ex.clicked.connect(self.export)
        mv.addWidget(self.btn_ex)
        for k in [self.k_att, self.k_rel, self.k_pan, self.k_verb, self.k_pt, self.k_tn, self.k_cr, self.k_cmp]:
            k.valueChanged.connect(self.auto_prev)
        self.t_rev.stateChanged.connect(self.auto_prev)
        self.ptimer = QTimer(interval=150, singleShot=True, timeout=self.prev)
        self.atimer = QTimer(interval=16, timeout=self.tick)

    def set_reg(self, r): self.c_reg = r; self.auto_prev()
    
    def auto_prev(self):
        self.ptimer.start(120)
        
    def get_p(self):
        return {'attack': self.k_att.value(), 'release': self.k_rel.value(),
                'pan': self.k_pan.value(),
                'reverse': self.t_rev.checked,
                'rate': self.k_pt.value(), 'tone': self.k_tn.value(),
                'crush': self.k_cr.value(), 
                'compress': self.k_cmp.value(),
                'verb': self.k_verb.value(), 
                'clicks': self.t_clk.checked,
                'repeat': self.t_rep.checked}

    def prev(self):
        # If we are already running, don't stack another thread on top.
        # Instead, signal that we need to run again as soon as this one is done.
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.update_pending = True
            self.worker_thread.abort()  # Ask it to stop early
            return

        # If we are free, start processing immediately
        self.start_processing()

    def start_processing(self):
        if self.data is None: return

        self.is_processing = True 
        self.update_pending = False 

        self.vol_anim.stop()
        self.vol_anim.setStartValue(self.ao.volume())
        self.vol_anim.setEndValue(0.0)
        self.vol_anim.start()

        p = self.get_p()

        # Include self.c_reg (region) in the hash key. 
        # Changing region invalidates the cache because the slice changes.
        current_hash = (p['tone'], p['crush'], p['compress'], self.c_reg)
        
        cached_source = None
        if self.dsp_cache is not None and self.dsp_hash == current_hash:
            cached_source = self.dsp_cache
            
        self.worker_thread = ExportThread(self.data, self.sr, self.c_reg, p, cached_source, current_hash)
        self.worker_thread.finished_ok.connect(self.on_fin)
        self.worker_thread.finished.connect(self.on_thread_finished)
        self.worker_thread.start()
    
    def on_thread_finished(self):
        # 1. Clean up the reference to the finished thread
        sender = self.sender()
        
        # Move to trash list to prevent "QThread destroyed while running" crash
        # (It keeps the Python object alive until C++ is truly done)
        if sender not in self.old_threads:
            self.old_threads.append(sender)
        
        # If this was our primary worker, clear the slot
        if sender == self.worker_thread:
            self.worker_thread = None

        # 2. Check if the user changed settings while we were working
        if self.update_pending:
            # Recursion: Start the next calculation immediately
            # This ensures the UI always eventually shows the latest state
            self.start_processing()
        else:
            # We are truly done. 
            self.is_processing = False
            
            # Clean up the trash list now that things are calm
            self.old_threads.clear()
    
    def cleanup_old_thread(self):
        # The sender is the thread that just finished
        sender = self.sender()
        if sender in self.old_threads:
            self.old_threads.remove(sender)
            # Now that it is removed from the list AND execution has finished, 
            # Python can safely Garbage Collect it.

    def on_fin(self, d, sr, g_map, new_cache):

        self.is_processing = False

        if new_cache is not None:
            self.dsp_cache = new_cache
            p = self.get_p()
            self.dsp_hash = (p['tone'], p['crush'], p['compress'])

        self.p_data = d 
        self.grain_map = g_map
        self.wave.set_grain_map(g_map) 
        
        try:
            fd, p = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            MicroEngine.save_file(p, d, sr)
            
            # Unload previous file to release lock
            self.player.stop()
            self.player.setSource(QUrl()) 
            
            if self.tf and os.path.exists(self.tf):
                try: os.remove(self.tf)
                except: pass
            self.tf = p
            
            # Restore volume and play
            self.vol_anim.stop()
            self.ao.setVolume(1.0) # Force volume back to 100%
            
            self.player.setSource(QUrl.fromLocalFile(p))
            self.player.play()
            self.atimer.start()
        except Exception as e: 
            print(f"Playback Error: {e}")

    def tick(self):
        if self.p_data is None or len(self.p_data) == 0: return

        state = self.player.playbackState()
        
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
            
            total_dur_ms = (len(self.p_data) / self.sr) * 1000.0
            if total_dur_ms > 0:
                current_loop_ms = est_pos % total_dur_ms
                self.wave.set_playback_pos(current_loop_ms)
                
                if self.t_rep.checked:
                    self.wave.set_play_head(-1) 
                else:
                    self.wave.set_play_head(current_loop_ms / total_dur_ms)
        else:
            self.wave.set_play_head(0.0)
            self.wave.set_playback_pos(-1)

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
        self.dsp_cache, self.dsp_hash = None, None
        self.wave.set_data(None)
        self.player.setSource(QUrl())

    def export(self):
        if self.p_data is None: return
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "Music", "micro")
        
        if not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except:
                self.btn_ex.setText("err: cannot create folder")
                return

        n = f"micro_export_{int(time.time())}.wav"
        full_path = os.path.join(save_dir, n)

        try: 
            MicroEngine.save_file(full_path, self.p_data, self.sr)
            self.header.intensify()
            self.btn_ex.setText("saved to Music/micro")
            QTimer.singleShot(2000, lambda: self.btn_ex.setText("export wav"))
        except: 
            self.btn_ex.setText("error saving")

    def dragEnterEvent(self, e): e.accept() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): self.load_p(e.mimeData().urls()[0].toLocalFile())
    def open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "open", "", "audio (*.wav *.mp3 *.flac *.ogg)")
        if p: self.load_p(p)
    def load_p(self, p):
        try:
            d, sr = MicroEngine.load_file(p); self.data, self.sr = d, sr
            self.dsp_cache = None 
            self.wave.set_data(d); self.prev()
        except: pass
    def closeEvent(self, e):
        if self.tf and os.path.exists(self.tf): 
            try: os.remove(self.tf)
            except: pass
        e.accept()

if __name__ == '__main__':
    try:
        import ctypes
        myappid = 'micro.audio.tool.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception: pass

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLES)
    
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    elif os.path.exists("micro.ico"):
        app.setWindowIcon(QIcon("micro.ico"))

    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
