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
        tail_len = int(sr * 0.4) 
        noise = np.random.randn(tail_len)
        env = np.exp(-np.linspace(0, 1, tail_len) * 7.0)
        ir = noise * env
        b, a = signal.butter(1, 0.4, 'low')
        ir = signal.lfilter(b, a, ir)
        wet = signal.fftconvolve(x, ir, mode='full')[:len(x)]
        wet = wet / (np.max(np.abs(wet)) + 1e-9)
        return (1 - mix) * x + mix * wet

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
    def process_grain(data, sr, region, params):
        is_repeat = params.get('repeat', False)
        
        # --- Helper: Cut audio at zero crossings ---
        def make_one_cut(sub_region):
            raw_start = int(sub_region[0] * len(data))
            raw_end = int(sub_region[1] * len(data))
            
            start_idx = MicroEngine.get_zero_crossing(data, raw_start, 1024)
            end_idx = MicroEngine.get_zero_crossing(data, raw_end, 1024)
            
            if start_idx >= end_idx: return np.zeros((1024, 2), dtype=np.float32), start_idx, start_idx
            
            chunk = data[start_idx:end_idx].copy()
            if len(chunk) == 0: return np.zeros((1024, 2), dtype=np.float32), start_idx, start_idx

            return chunk, start_idx, end_idx

        # --- Helper: Apply Full FX Chain ---
        def apply_fx(chunk):
            # 1. Fades (smoother for overlap)
            fade_len = min(int(sr * 0.05), len(chunk) // 2) 
            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                if chunk.ndim > 1:
                    fade_in = fade_in[:, None]
                    fade_out = fade_out[:, None]
                chunk[:fade_len] *= fade_in
                chunk[-fade_len:] *= fade_out

            # 2. Reverse
            if params.get('reverse', False): chunk = chunk[::-1]

            # 3. Rate / Resample
            rate = params.get('rate', 1.0)
            if abs(rate - 1.0) > 0.01:
                new_len = int(len(chunk) / rate)
                if new_len > 0: chunk = signal.resample(chunk, new_len)

            # 4. Bitcrush
            crush = params.get('crush', 0.0)
            if crush > 0.01:
                factor = 1.0 - (crush * 0.98) 
                target_len = int(len(chunk) * factor)
                if target_len > 5:
                    down = signal.resample(chunk, target_len)
                    chunk = signal.resample(down, len(chunk))

            # 5. Tone (Filter)
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

            # 6. Compressor
            comp = params.get('compress', 0.0)
            if comp > 0.01:
                sos_env = signal.butter(1, 0.02, output='sos')
                det_sig = np.mean(np.abs(chunk), axis=1) if chunk.ndim > 1 else np.abs(chunk)
                env = signal.sosfilt(sos_env, det_sig)
                thresh = 0.6 * (1.0 - comp * 0.7)
                gain_red = np.minimum(1.0, thresh / (env + 1e-6))
                if chunk.ndim > 1: gain_red = gain_red[:, None]
                chunk = chunk * gain_red * (1.0 + comp * 2.5)

            # 7. Envelope (Attack/Release)
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
            
            # 8. Reverb
            v_amt = params.get('verb', 0.0)
            if v_amt > 0.01: chunk = MicroEngine.apply_reverb(chunk, sr, v_amt * 0.6)

            # 9. Clicks/Glitch
            if params.get('clicks', False):
                c_len = min(len(chunk), 2000) 
                if c_len > 100:
                    raw_src = chunk if chunk.ndim == 1 else chunk[:, 0]
                    raw = raw_src[100:c_len][::2]
                    click_sig = np.diff(raw, prepend=0) 
                    max_val = np.max(np.abs(click_sig))
                    if max_val > 1e-5: click_sig = click_sig / max_val
                    c_env = np.exp(-np.linspace(0, 15, len(click_sig)))
                    click_sig = click_sig * c_env * 2.5 
                    if chunk.ndim > 1: click_sig = np.column_stack((click_sig, click_sig))
                    grid_step = int(sr * 0.25)
                    for pos in range(0, len(chunk), grid_step):
                        if random.random() < 0.35:
                            remaining = len(chunk) - pos
                            write_len = min(len(click_sig), remaining)
                            if write_len > 0:
                                segment = chunk[pos:pos+write_len] + click_sig[:write_len]
                                chunk[pos:pos+write_len] = np.clip(segment, -1.0, 1.0)

            # 10. Stereo & Pan
            if chunk.ndim == 1: chunk = np.column_stack((chunk, chunk))
            
            pan_depth = params.get('pan', 0.0)
            if pan_depth > 0.01:
                # "Symphonic" wide random pan
                pos_rng = random.uniform(-1.0, 1.0)
                width = pan_depth * 0.8 
                l_gain = 1.0 - (width * (0.5 + 0.5 * pos_rng))
                r_gain = 1.0 - (width * (0.5 + 0.5 * -pos_rng))
                chunk[:, 0] *= l_gain
                chunk[:, 1] *= r_gain
            
            return np.tanh(chunk)

        grain_map = [] 
        
        if not is_repeat:
            # Single Shot
            chunk, s_idx, e_idx = make_one_cut(region)
            if len(chunk) == 0: return np.zeros((1024, 2), dtype=np.float32), []
            processed = apply_fx(chunk)
            grain_map.append((0, (len(processed)/sr)*1000.0, region[0], region[1]))
            return np.clip(processed, -0.99, 0.99), grain_map
        else:
            # --- SYMPHONIC POLYPHONIC REPEAT ---
            bpm = random.randint(90, 120)
            beat_sec = 60.0 / bpm
            step_16_sec = beat_sec / 4.0
            
            # 2 Bars
            total_slots = 32 
            seq_len_samples = int(total_slots * step_16_sec * sr)
            seq_buffer = np.zeros((seq_len_samples, 2), dtype=np.float32)
            
            # HIGH DENSITY: 12 to 24 grains
            count = random.randint(12, 24)
            active_slots = sorted([random.randint(0, total_slots-1) for _ in range(count)])
            
            reg_len = region[1] - region[0]
            safe_padding = 0.001

            for slot_idx in active_slots:
                # Source
                effective_len = max(0, reg_len - safe_padding)
                r_offset = random.uniform(0, effective_len)
                s_pt = region[0] + r_offset
                
                # Long duration for overlap
                dur_factor = random.uniform(2.0, 8.0) 
                
                max_len_norm = region[1] - s_pt
                desired_len_samples = int(step_16_sec * dur_factor * sr)
                desired_len_norm = desired_len_samples / len(data)
                actual_len_norm = min(desired_len_norm, max_len_norm)
                e_pt = s_pt + actual_len_norm
                
                raw_chunk, s_idx, e_idx = make_one_cut((s_pt, e_pt))
                if len(raw_chunk) == 0: continue
                
                grain = apply_fx(raw_chunk)
                
                # Mix
                insert_idx = int(slot_idx * step_16_sec * sr)
                write_len = min(len(grain), seq_len_samples - insert_idx)
                
                if write_len > 0:
                    seq_buffer[insert_idx : insert_idx + write_len] += grain[:write_len]
                    
                    # Map
                    p_start_ms = (insert_idx / sr) * 1000.0
                    p_end_ms = ((insert_idx + write_len) / sr) * 1000.0
                    src_s_norm = s_idx / len(data)
                    src_e_norm = (s_idx + write_len) / len(data)
                    
                    if params.get('reverse', False):
                        grain_map.append((p_start_ms, p_end_ms, src_e_norm, src_s_norm))
                    else:
                        grain_map.append((p_start_ms, p_end_ms, src_s_norm, src_e_norm))

            # Soft Limiter
            peak = np.max(np.abs(seq_buffer))
            if peak > 0.9:
                seq_buffer = seq_buffer * (0.9 / peak)
            
            return seq_buffer, grain_map

class ExportThread(QThread):
    finished_ok = pyqtSignal(object, int, list)
    def __init__(self, data, sr, region, params):
        super().__init__()
        self.data, self.sr, self.region, self.params = data, sr, region, params
    def run(self):
        try:
            processed, g_map = MicroEngine.process_grain(self.data, self.sr, self.region, self.params)
            self.finished_ok.emit(processed, self.sr, g_map)
        except Exception as e: 
            print(e)

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
        
        # Granular State
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
        
        # Pan Check
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

        # Draw Selection
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

        # --- MULTI-HEAD VISUALIZATION (Taller & Subtle) ---
        if self.grain_map and self.current_loop_ms >= 0:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            for (p_start, p_end, src_start, src_end) in self.grain_map:
                if p_start <= self.current_loop_ms < p_end:
                    
                    # Progress 0.0 -> 1.0
                    progress = (self.current_loop_ms - p_start) / (p_end - p_start)
                    
                    # Current Source Position
                    cur_src = src_start + (src_end - src_start) * progress
                    gx = (cur_src - self.view_offset) / vw * w
                    
                    if 0 <= gx <= w:
                        # Subtle Pastel Color (Low Alpha)
                        hue = (cur_src * 5.0 + self.hue_anim) % 1.0
                        # Alpha 0.35 (88/255) for overlapping subtlety
                        col = QColor.fromHslF(hue, 0.6, 0.75, 0.35) 
                        
                        painter.setBrush(col)
                        # Very faint stroke
                        painter.setPen(QPen(QColor(255, 255, 255, 100), 0.5))
                        
                        # Tall Triangle Geometry
                        # Base anchored at bottom (h)
                        tri_h = 55  
                        tri_w = 8   # Thinner (was 16)
                        y_base = h  # Pixel-aligned bottom
                        
                        tri = QPolygonF([
                            QPointF(gx, y_base - tri_h),      # Tip (Up)
                            QPointF(gx - tri_w/2, y_base),    # Bottom Left
                            QPointF(gx + tri_w/2, y_base)     # Bottom Right
                        ])
                        painter.drawPolygon(tri)

        # --- Standard Playhead (Fallback) ---
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
        self.resize(650, 520)
        self.setAcceptDrops(True)
        self.data, self.sr, self.p_data, self.tf = None, 44100, None, None
        self.grain_map = []
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
        if not self.wave.is_dragging: self.ptimer.start()
        
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
        if self.data is None: return
        self.player.stop()
        self.th = ExportThread(self.data, self.sr, self.c_reg, self.get_p())
        self.th.finished_ok.connect(self.on_fin)
        self.th.start()

    def on_fin(self, d, sr, g_map): 
        self.p_data = d 
        self.grain_map = g_map
        self.wave.set_grain_map(g_map) # Pass map to visualizer
        try:
            fd, p = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            MicroEngine.save_file(p, d, sr)
            self.tf = p
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(p))
            self.player.play()
            self.atimer.start()
        except: pass

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
                
                # 1. Feed exact loop time to visualizer
                self.wave.set_playback_pos(current_loop_ms)
                
                # 2. Handle Playhead fallback
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
        self.wave.set_data(None)
        self.player.setSource(QUrl())

    def export(self):
        if self.p_data is None: return
        n = f"micro_export_{int(time.time())}.wav"
        try: 
            MicroEngine.save_file(os.path.join(os.getcwd(), n), self.p_data, self.sr)
            self.header.intensify()
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