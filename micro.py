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
        
        tail_len = int(sr * 1.0) 
        noise = np.random.randn(tail_len)
        env = np.exp(-np.linspace(0, 1, tail_len) * 8.0)
        ir = noise * env
        
        b, a = signal.butter(1, 300 / (sr/2), 'high')
        ir = signal.lfilter(b, a, ir)
        
        padding = np.zeros((tail_len, x.shape[1] if x.ndim > 1 else 1), dtype=np.float32)
        if x.ndim == 1: x = x[:, None]
        padded_x = np.concatenate([x, padding])
        
        wet = signal.fftconvolve(padded_x, ir[:, None] if ir.ndim==1 else ir, mode='full')
        wet = wet[:len(padded_x)]
        wet = wet / (np.max(np.abs(wet)) + 1e-9)
        
        dry_lvl = 1.0 - (mix * 0.15) 
        wet_lvl = mix * 0.35
        out = (padded_x * dry_lvl) + (wet * wet_lvl)
        return np.tanh(out)

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
    def apply_lofi(data, factor_0_to_1):
        if factor_0_to_1 <= 0.01: return data
        step = int(1 + factor_0_to_1 * 4)
        if step > 1:
            down = data[::step]
            if data.ndim > 1:
                up = np.repeat(down, step, axis=0)
            else:
                up = np.repeat(down, step)
            if len(up) > len(data):
                up = up[:len(data)]
            elif len(up) < len(data):
                diff = len(data) - len(up)
                padding = np.tile(up[-1:], (diff, 1)) if data.ndim > 1 else np.tile(up[-1:], diff)
                up = np.concatenate([up, padding])
            return up
        return data

    @staticmethod
    def precompute_heavy_fx(data, sr, params):
        chunk = data.copy()
        crush = params.get('crush', 0.0)
        chunk = MicroEngine.apply_lofi(chunk, crush)

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
        return chunk

    @staticmethod
    def render_playback(source_data, sr, region, params, abort_check=None): 
        grain_val = params.get('grain', 0.0)
        is_granular = grain_val > 0.05
        grain_map = []
        
        # --- PROCESS SLICE (No Clicks Here) ---
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
        
        if not is_granular:
            if abort_check and abort_check(): return None, [] 
            chunk, s, e = process_slice(region)
            final_buffer = chunk
            grain_map.append((0, (len(chunk)/sr)*1000.0, region[0], region[1]))
        else:
            bpm = random.randint(90, 120)
            beat_sec = 60.0 / bpm
            slot_dur = beat_sec / 4.0 
            total_slots = 32
            seq_len = int(total_slots * slot_dur * sr)
            seq_buffer = np.zeros((seq_len, 2), dtype=np.float32)
            
            # Continuous grain generation - overlapping chain
            density = int(8 + grain_val * 40)  # Overall density
            
            # Start with initial grain
            current_time = 0.0
            
            # Track grain timings for crossfade at loop point
            grain_timings = []
            
            # Generate grains continuously, not limited by density for the initial pass
            # We'll generate enough to fill the buffer with overlap
            max_time = total_slots * slot_dur
            generated_grains = 0
            
            while current_time < max_time:
                if abort_check and abort_check(): return None, []
                
                # Grain duration - ensure continuity by overlapping
                dur = random.uniform(0.1, 0.25)  # Longer grains for continuity
                
                # Position within source region
                s_pt = region[0] + random.uniform(0, region[1] - region[0] - 0.01)
                len_norm = dur / (region[1] - region[0]) * 1.5
                e_pt = min(region[1], s_pt + len_norm)
                
                if e_pt - s_pt < 0.001: continue
                
                grain, s_idx, e_idx = process_slice((s_pt, e_pt))
                if len(grain) == 0: continue
                
                ins_idx = int(current_time * sr)
                w_len = min(len(grain), seq_len - ins_idx)
                if w_len > 0:
                    # Store grain timing for crossfade calculation
                    grain_timings.append((ins_idx, ins_idx + w_len))
                    
                    # Apply smooth envelope - use longer fade for better overlap
                    env_len = min(w_len * 2, len(grain))
                    env = np.hanning(env_len)[:w_len]
                    
                    # Make envelope more gentle for better overlapping
                    env = np.power(env, 0.5)  # Even flatter envelope
                    
                    if grain.ndim > 1:
                        grain_slice = grain[:w_len].copy() * env[:, None]
                    else:
                        grain_slice = grain[:w_len].copy() * env
                    
                    seq_buffer[ins_idx:ins_idx+w_len] += grain_slice * 0.6
                    
                    # Add to grain map
                    p_s = (ins_idx/sr)*1000.0
                    p_e = ((ins_idx+w_len)/sr)*1000.0
                    n_s = s_idx/len(source_data)
                    n_e = (s_idx+w_len)/len(source_data)
                    grain_map.append((p_s, p_e, n_s, n_e))
                    
                    generated_grains += 1

                base_ov = 0.1 + (grain_val * 0.8) 
                overlap_amount = random.uniform(base_ov, min(0.95, base_ov + 0.1))
                current_time += dur * (1.0 - overlap_amount)
            
            # Create seamless loop by crossfading end to beginning
            # Find grains that cross the loop boundary
            loop_crossfade_time = 0.05  # 50ms crossfade at loop point
            crossfade_samples = int(loop_crossfade_time * sr)
            
            if crossfade_samples > 0:
                # Apply fade in to the beginning
                fade_in = np.linspace(0, 1, crossfade_samples)
                if seq_buffer.ndim > 1:
                    seq_buffer[:crossfade_samples] *= fade_in[:, None]
                else:
                    seq_buffer[:crossfade_samples] *= fade_in
                
                # Apply fade out to the end
                fade_out = np.linspace(1, 0, crossfade_samples)
                if seq_buffer.ndim > 1:
                    seq_buffer[-crossfade_samples:] *= fade_out[:, None]
                else:
                    seq_buffer[-crossfade_samples:] *= fade_out
                
                # Now crossfade: mix the end of buffer into beginning
                # Get the audio from the end that will crossfade into beginning
                end_audio = seq_buffer[-crossfade_samples:].copy()
                
                # Apply fade-in to the end audio (so it starts quiet at loop point)
                if end_audio.ndim > 1:
                    end_audio *= fade_in[:, None]
                else:
                    end_audio *= fade_in
                
                # Mix it with the beginning audio (which already has fade-in applied)
                seq_buffer[:crossfade_samples] += end_audio * 0.8  # Reduced gain to prevent buildup
            
            # Alternative approach: Ensure grains wrap around seamlessly
            # Generate additional grains that start before the end and continue past it
            if generated_grains > 0:
                # Find the last grain
                last_grain_end = grain_timings[-1][1] if grain_timings else 0
                buffer_end_time = seq_len / sr
                
                # If there's space at the end, generate a grain that starts near the end
                # and another that starts at the beginning to overlap
                if last_grain_end < seq_len:
                    # Time remaining in buffer
                    remaining_time = (seq_len - last_grain_end) / sr
                    
                    if remaining_time > 0.05:  # If there's significant time remaining
                        # Generate a grain that starts near the end
                        start_time = buffer_end_time - random.uniform(0.1, 0.3)
                        if start_time > 0:
                            # Generate grain as before
                            dur = random.uniform(0.1, 0.25)
                            s_pt = region[0] + random.uniform(0, region[1] - region[0] - 0.01)
                            len_norm = dur / (region[1] - region[0]) * 1.5
                            e_pt = min(region[1], s_pt + len_norm)
                            
                            if e_pt - s_pt > 0.001:
                                grain, s_idx, e_idx = process_slice((s_pt, e_pt))
                                if len(grain) > 0:
                                    ins_idx = int(start_time * sr)
                                    w_len = min(len(grain), seq_len - ins_idx)
                                    if w_len > 0:
                                        # This grain will get cut off at buffer end
                                        env = np.hanning(w_len * 2)[:w_len]
                                        env = np.power(env, 0.5)
                                        
                                        if grain.ndim > 1:
                                            grain_slice = grain[:w_len].copy() * env[:, None]
                                        else:
                                            grain_slice = grain[:w_len].copy() * env
                                        
                                        seq_buffer[ins_idx:ins_idx+w_len] += grain_slice * 0.6
            
            # Final normalization with careful headroom
            pk = np.max(np.abs(seq_buffer))
            if pk > 0.85:  # More headroom for overlapped grains
                seq_buffer *= (0.85/pk)
            
            final_buffer = seq_buffer
        
        click_amt = params.get('clicks', 0.0)
        if click_amt > 0.01 and final_buffer is not None:
            c_bpm = random.randint(90, 150)
            divs = [0.5, 0.25, 0.125]  # More rhythmic variety
            step_len = (60.0/c_bpm) * random.choice(divs)
            grid_sz = int(step_len * sr)
            
            prob = 0.05 + (click_amt * 0.3)  # Lower probability
            last_end_pos = 0
            crush_amt = params.get('crush', 0.0)
            
            buffer_len = len(final_buffer)
            
            for pos in range(0, buffer_len, grid_sz):
                if pos > last_end_pos and random.random() < prob:
                    dur_samps = int((random.uniform(1.0, 4.0) / 1000.0) * sr)
                    
                    if pos + dur_samps < buffer_len:
                        # --- Lower Tone Chance & Higher Volume ---
                        # 30% chance for low thud, else high click
                        if random.random() < 0.3:
                            freq = random.uniform(60, 300)
                        else:
                            freq = random.uniform(800, 22000)
                            
                        t = np.arange(dur_samps) / sr
                        
                        # Original amplitude - more audible
                        tone_wave = np.sin(2 * np.pi * freq * t)
                        tone_wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
                        
                        c_env = np.power(np.linspace(1, 0, dur_samps), 3.0)
                        
                        # Louder clicks
                        burst = tone_wave * c_env * (0.15 + click_amt * 0.25)
                        
                        if crush_amt > 0.01: 
                            burst = MicroEngine.apply_lofi(burst, crush_amt)
                        
                        if final_buffer.ndim > 1: 
                            pan = random.uniform(-0.3, 0.3)
                            left_gain = 1.0 - max(0, pan)
                            right_gain = 1.0 - max(0, -pan)
                            final_buffer[pos:pos+dur_samps, 0] += burst * left_gain
                            final_buffer[pos:pos+dur_samps, 1] += burst * right_gain
                        else: 
                            final_buffer[pos:pos+dur_samps] += burst
                            
                        last_end_pos = pos + dur_samps

        # 2. SHAKE
        shake_amt = params.get('shake', 0.0)
        if shake_amt > 0.01 and final_buffer is not None:
            ln = len(final_buffer)
            points = max(5, int(ln / sr * (5 + shake_amt * 15))) 
            noise_ctrl = np.random.uniform(1.0 - shake_amt, 1.0, points)
            shake_env = signal.resample(noise_ctrl, ln)
            if final_buffer.ndim > 1:
                final_buffer *= shake_env[:, None]
            else:
                final_buffer *= shake_env

        # 3. REVERB
        v_amt = params.get('verb', 0.0)
        if v_amt > 0.01:
            if abort_check and abort_check(): return None, []
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
        # Generate dots as normalized coordinates (0.0 to 1.0)
        cw, ch = 600, 34 
        cx, cy = cw/2, ch/2
        max_dist = math.sqrt(cx*cx + cy*cy)
        for i in range(250):
            x = random.randint(0, cw)
            y = random.randint(0, ch)
            dist = math.sqrt((x-cx)**2 + (y-cy)**2) / max_dist
            prob = 1.0 - (dist * 0.8)
            if random.random() < prob:
                # Store x, y as ratios of the container size
                self.dots.append([x/cw, y/ch, random.randint(1, 2), random.random(), random.random()*6.28])
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
            nx, ny, sz, hue, off = d
            # Calculate actual position based on current window size
            x = nx * w
            y = ny * h
            
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
        
        # Font and Color adjustments
        txt_col = QColor("#64748b") 
        txt_col.setAlpha(180) 
        if self.checked: 
            # Less blue, more neutral slate gray
            txt_col = QColor("#475569")
            txt_col.setAlpha(240)
            
        p.setPen(txt_col)
        # Weight changed from Bold to Medium (Weight 500)
        font = QFont("Segoe UI", 9, QFont.Weight.Medium)
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
        
        # Font and Color adjustments
        txt_col = QColor("#64748b")
        txt_col.setAlpha(180) 
        p.setPen(txt_col)
        # Weight changed from Bold to Medium
        font = QFont("Segoe UI", 9, QFont.Weight.Medium)
        p.setFont(font)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.label)

class MorphPlayButton(QWidget):
    clicked = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._state = 0 
        self._hover = False
        self.anim_val = 0.0 
        self.icon_morph = 0.0 
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)

    def set_state(self, s):
        self._state = s
        self.update()

    def mousePressEvent(self, e): self.clicked.emit()
    def enterEvent(self, e): self._hover = True
    def leaveEvent(self, e): self._hover = False

    def animate(self):
        target_anim = 0.0 if self._state == 0 else 1.0
        self.anim_val += (target_anim - self.anim_val) * 0.25
        
        target_morph = 1.0 if self._state == 1 else 0.0
        self.icon_morph += (target_morph - self.icon_morph) * 0.25
        
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        
        # Use QRectF center for float precision
        c = QRectF(r).center()
        
        # Background: Constant color (No hover darken)
        bg = QColor("#e2e8f0")
        p.setBrush(bg)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 12, 12)
        
        # Foreground
        fg = QColor("#64748b")
        fg.setAlpha(180) 
        p.setBrush(fg)
        
        # Dot (Idle State)
        dot_sz = 4.0 * (1.0 - self.anim_val)
        if dot_sz > 0.1:
            p.drawEllipse(c, dot_sz, dot_sz)
            
        # Icon Morph
        if self.anim_val > 0.01:
            scale = self.anim_val
            p.translate(c)
            p.scale(scale, scale)
            t = self.icon_morph
            
            # Geometry
            p1x = -3.0 * (1.0-t) + (-5.0) * t
            p1y = -5.0 
            p2x = -3.0 * (1.0-t) + (-5.0) * t
            p2y = 5.0 
            p3x = 6.0 * (1.0-t) + (-2.0) * t
            p3y = 0.0 * (1.0-t) + (5.0) * t
            p4x = 6.0 * (1.0-t) + (-2.0) * t
            p4y = 0.0 * (1.0-t) + (-5.0) * t
            
            path = QPainterPath()
            path.moveTo(p1x, p1y)
            path.lineTo(p2x, p2y)
            path.lineTo(p3x, p3y)
            path.lineTo(p4x, p4y)
            path.closeSubpath()
            p.drawPath(path)
            
            # Right Pause Bar
            if t > 0.01:
                current_alpha = int(180 * t)
                fg.setAlpha(current_alpha)
                p.setBrush(fg)
                
                off = (1.0 - t) * 1.5
                p.drawRect(QRectF(2.0 + off, -5.0, 3.0, 10.0))

class PastelExportButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.phase = 0.0
        self.hover_anim = 0.0 # New interpolation value
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20) # Faster tick for smooth animation
        self.hovered = False

    def animate(self):
        self.phase = (self.phase + 0.01) % 1.0
        
        # Smooth interpolation of hover state
        target = 1.0 if self.hovered else 0.0
        diff = target - self.hover_anim
        if abs(diff) > 0.01:
            self.hover_anim += diff * 0.15 # Ease speed
            self.update()
        elif self.hovered: 
            self.update() # Keep updating for phase shift if hovered

    def enterEvent(self, e): self.hovered = True; self.update()
    def leaveEvent(self, e): self.hovered = False; self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        grad = QLinearGradient(0, 0, rect.width(), 0)
        
        # Interpolate saturation and lightness based on hover_anim
        base_s, base_l = 0.4, 0.94
        hover_s, hover_l = 0.6, 0.85
        
        cur_s = base_s + (hover_s - base_s) * self.hover_anim
        cur_l = base_l + (hover_l - base_l) * self.hover_anim
        
        for i in range(3):
            t = i / 2.0
            h = (self.phase + t * 0.2) % 1.0
            grad.setColorAt(t, QColor.fromHslF(h, cur_s, cur_l))
            
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
        # 0.002 instead of 0.005
        self.hue_anim = (self.hue_anim + 0.002) % 1.0
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
        last_mode = self.drag_mode 
        
        self.is_dragging = False
        self.drag_mode = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Do not trigger processing if we were just panning
        if last_mode == 'pan':
            return 

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

        # RICHER PASTEL GRADIENT (Better visibility for pulse)
        grad_wave = QLinearGradient(0, 0, w, 0)
        grad_wave.setColorAt(0.0, QColor("#ff9aa2")) # Richer Pastel Red
        grad_wave.setColorAt(0.35, QColor("#e0b0ff")) # Richer Lilac
        grad_wave.setColorAt(0.7, QColor("#b5b9ff")) # Richer Periwinkle
        grad_wave.setColorAt(1.0, QColor("#97c2fc")) # Richer Blue
        
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
        
        # 1. Background with VISIBLE ETHEREAL PULSE
        # Speed: 8x faster (was too slow). Depth: 0.6 to 1.0 (wider range).
        pulse_opacity = 0.8 + 0.2 * math.sin(self.hue_anim * 8 * math.pi)
        
        painter.save()
        painter.setOpacity(pulse_opacity)
        painter.drawPixmap(0, 0, self.cached_px)
        painter.restore()

        if self.data is None: return

        vw = 1.0 / self.zoom_level
        s = min(self.sel_start, self.sel_end)
        e = max(self.sel_start, self.sel_end)
        sx = (s - self.view_offset) / vw * w
        ex = (e - self.view_offset) / vw * w
        sel_w = ex - sx

        # 2. Draw Grains
        if self.grain_map and self.current_loop_ms >= 0:
            painter.save()
            if sel_w > 0:
                painter.setClipRect(QRectF(sx, 0, sel_w, float(h)))
            
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            show_trails = len(self.grain_map) > 5
            fade_window = 150.0 
            
            for (p_start, p_end, src_start, src_end) in self.grain_map:
                if p_start <= self.current_loop_ms < (p_end + fade_window):
                    is_dying = self.current_loop_ms >= p_end
                    
                    if is_dying:
                        progress = 1.0
                        death_prog = (self.current_loop_ms - p_end) / fade_window
                        base_alpha = max(0, int(100 * (1.0 - death_prog)))
                    else:
                        progress = (self.current_loop_ms - p_start) / (p_end - p_start)
                        fade_curve = math.sin(progress * math.pi)
                        base_alpha = 100 + int(155 * fade_curve)

                    cur_src = src_start + (src_end - src_start) * progress
                    gx = (cur_src - self.view_offset) / vw * w
                    
                    if gx < sx - 50 or gx > ex + 50: continue

                    hue = (cur_src * 2.0 + self.hue_anim) % 1.0
                    tri_h = 24  
                    tri_w = 12  
                    y_base = float(h)
                    
                    if show_trails and not is_dying:
                        trail_lag = 0.05
                        if progress > trail_lag:
                            trail_prog = progress - trail_lag
                            trail_src = src_start + (src_end - src_start) * trail_prog
                            tx = (trail_src - self.view_offset) / vw * w
                            
                            if abs(gx - tx) < w * 0.15:
                                trail_grad = QLinearGradient(gx, y_base, tx, y_base)
                                c_start = QColor.fromHslF(hue, 0.6, 0.8, base_alpha/255.0 * 0.6) 
                                c_end = QColor.fromHslF(hue, 0.6, 0.8, 0.0)
                                trail_grad.setColorAt(0.0, c_start)
                                trail_grad.setColorAt(1.0, c_end)
                                
                                painter.setBrush(QBrush(trail_grad))
                                painter.setPen(Qt.PenStyle.NoPen)
                                
                                trail_poly = QPolygonF([
                                    QPointF(gx, y_base - tri_h + 5),
                                    QPointF(gx, y_base),
                                    QPointF(tx, y_base),
                                    QPointF(tx, y_base - 2)
                                ])
                                painter.drawPolygon(trail_poly)

                    t_grad = QLinearGradient(gx, y_base, gx, y_base - tri_h)
                    col_btm = QColor.fromHslF(hue, 0.6, 0.75, base_alpha/255.0)
                    col_top = QColor.fromHslF(hue, 0.6, 0.85, (base_alpha*0.5)/255.0)
                    t_grad.setColorAt(0.0, col_btm)
                    t_grad.setColorAt(1.0, col_top)
                    
                    painter.setBrush(QBrush(t_grad))
                    painter.setPen(Qt.PenStyle.NoPen)
                    
                    tri = QPolygonF([
                        QPointF(gx, y_base - tri_h),
                        QPointF(gx - tri_w/2, y_base),
                        QPointF(gx + tri_w/2, y_base)
                    ])
                    painter.drawPolygon(tri)
            
            painter.restore()

        # 3. Selection Overlay
        if sel_w > 1:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(63, 108, 155, 5))
            painter.drawRect(QRectF(sx, 0, sel_w, float(h)))
            
            grad_h = QLinearGradient(0, 0, 0, float(h))
            h1 = self.hue_anim
            h2 = (self.hue_anim + 0.3) % 1.0
            grad_h.setColorAt(0, QColor.fromHslF(h1, 0.5, 0.8))
            grad_h.setColorAt(1, QColor.fromHslF(h2, 0.5, 0.8))
            brush = QBrush(grad_h)
            
            painter.fillRect(QRectF(sx, 0, 3, float(h)), brush)
            painter.fillRect(QRectF(ex-3, 0, 3, float(h)), brush)

        # 4. Main Playhead
        if self.play_head >= 0 and not self.grain_map:
            ph_abs = s + self.play_head * (e - s)
            px = (ph_abs - self.view_offset) / vw * w
            
            if px <= ex:
                grad_ph = QLinearGradient(0, 0, 0, float(h))
                grad_ph.setColorAt(0.0, QColor(160, 190, 255, 200))
                grad_ph.setColorAt(1.0, QColor(255, 180, 200, 200))
                painter.setBrush(QBrush(grad_ph))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(px - 1.25, 0, 2.5, float(h)))

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
        self.worker_thread = None      
        self.update_pending = False    
        self.old_threads = []          
        self.resize(640, 480)
        self.setAcceptDrops(True)
        self.data, self.sr, self.p_data, self.tf = None, 44100, None, None
        self.is_processing = False
        self.update_pending = False
        
        self.dsp_cache = None
        self.dsp_hash = None 
        
        self.grain_map = []
        self.c_reg = (0.3, 0.7)
        self.player = QMediaPlayer()
        
        self.ao = QAudioOutput()
        self.player.setAudioOutput(self.ao)
        self.player.mediaStatusChanged.connect(self.media_status)
        
        self.vol_anim = QPropertyAnimation(self.ao, b"volume")
        self.vol_anim.setDuration(250)
        self.vol_anim.setEndValue(0.0)
        self.vol_anim.setEasingCurve(QEasingCurve.Type.Linear)
       
        self.last_media_pos = -1
        self.last_update_time = 0
        self.smooth_ms = 0.0 # NEW: High-precision accumulator
        
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
        
        # --- CONTROLS LAYOUT ---
        crow = QHBoxLayout(); crow.setSpacing(30)
        
        # Column 1 (Left)
        c1 = QVBoxLayout(); c1.setSpacing(4)
        self.k_att = ControlRow("attack", 0, 0.5, 0.01)
        self.k_rel = ControlRow("release", 0, 0.5, 0.1)
        self.k_pan = ControlRow("pan", 0.0, 1.0, 0.0) 
        self.k_verb = ControlRow("reverb", 0, 1, 0.0)
        self.k_clk = ControlRow("click", 0, 1, 0.0) # NEW CLICK SLIDER
        for w in [self.k_att, self.k_rel, self.k_pan, self.k_verb, self.k_clk]: c1.addWidget(w)
        c1.addStretch()
        
        # Column 2 (Right)
        c2 = QVBoxLayout(); c2.setSpacing(4)
        self.k_pt = ControlRow("rate", 0.5, 2.0, 1.0, "{:.2f}x")
        self.k_tn = ControlRow("tone", 0, 1, 0.5) 
        self.k_cr = ControlRow("crush", 0, 1, 0.0)
        self.k_shk = ControlRow("shake", 0, 1, 0.0) # NEW SHAKE SLIDER
        self.k_grn = ControlRow("grain", 0, 1, 0.0) # NEW GRAIN SLIDER
        for w in [self.k_pt, self.k_tn, self.k_cr, self.k_shk, self.k_grn]: c2.addWidget(w)
        c2.addStretch()
        
        crow.addLayout(c1, 1); crow.addLayout(c2, 1)
        mv.addLayout(crow)
        
        # --- TOGGLES ROW ---
        t_row = QHBoxLayout()
        t_row.addStretch() 
        
        self.btn_clear = PastelPush("clear")
        self.btn_clear.clicked.connect(self.clear_state)
        t_row.addWidget(self.btn_clear)
        
        # Reduced gap: 10 -> 4
        t_row.addSpacing(4)
        
        self.btn_play = MorphPlayButton()
        self.btn_play.clicked.connect(self.toggle_playback)
        t_row.addWidget(self.btn_play)
        
        # Reduced gap: 10 -> 4
        t_row.addSpacing(4)
        
        self.t_rev = PastelToggle("reverse")
        self.t_rev.stateChanged.connect(self.auto_prev)
        t_row.addWidget(self.t_rev)
        
        t_row.addStretch()
        
        mv.addLayout(t_row)
        
        self.btn_ex = PastelExportButton("export")
        self.btn_ex.clicked.connect(self.export)
        mv.addWidget(self.btn_ex)
        
        # Connections
        for k in [self.k_att, self.k_rel, self.k_pan, self.k_verb, self.k_clk, 
                  self.k_pt, self.k_tn, self.k_cr, self.k_shk, self.k_grn]:
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
                'shake': self.k_shk.value(),
                'verb': self.k_verb.value(), 
                'clicks': self.k_clk.value(), # Now a float 0-1
                'grain': self.k_grn.value()}  # Now a float 0-1

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

        # Update Hash: crush, tone stay. Shake is post-fx (render), so it doesn't need to be in the heavy_fx hash.
        current_hash = (p['tone'], p['crush'], self.c_reg)
        
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
            self.dsp_hash = (p['tone'], p['crush'], self.c_reg)

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
            self.ao.setVolume(1.0) 
            
            self.player.setSource(QUrl.fromLocalFile(p))

            # --- FIX START: Native Looping ---
            # Use native backend looping for gapless granular playback
            if self.k_grn.value() > 0.05:
                self.player.setLoops(QMediaPlayer.Loops.Infinite)
            else:
                self.player.setLoops(QMediaPlayer.Loops.Once)
            # --- FIX END ---

            self.player.play()
            self.btn_play.set_state(1) # Set to Playing
            self.atimer.start()
        except Exception as e: 
            print(f"Playback Error: {e}")

    def tick(self):
        if self.p_data is None or len(self.p_data) == 0: 
            return

        state = self.player.playbackState()
        
        if state == QMediaPlayer.PlaybackState.PlayingState:
            now = time.time()
            total_dur_ms = (len(self.p_data) / self.sr) * 1000.0
            if total_dur_ms <= 0: return

            if self.last_media_pos == -1:
                # Initialization
                raw_pos = self.player.position()
                self.smooth_ms = float(raw_pos)
                self.last_update_time = now
                self.last_media_pos = 0 
            else:
                # --- FREEWHEELING CLOCK ---
                # Rely on system time for smoothness, ignoring audio jitter
                dt = (now - self.last_update_time) * 1000.0
                self.last_update_time = now
                self.smooth_ms += dt
                
                # Handle Loop Wrapping
                if self.smooth_ms >= total_dur_ms:
                    self.smooth_ms %= total_dur_ms

                # --- FAILSAFE SYNC ---
                # Check for massive drift (e.g. Loop Reset or CPU Hang)
                raw_pos = self.player.position()
                
                # Calculate circular distance (shortest path around the loop)
                dist = raw_pos - self.smooth_ms
                if dist < -total_dur_ms / 2: dist += total_dur_ms
                elif dist > total_dur_ms / 2: dist -= total_dur_ms
                
                # Only hard snap if we are off by > 300ms (Massive drift)
                # This prevents micro-stutters from aggressive syncing.
                if abs(dist) > 300.0:
                    self.smooth_ms = float(raw_pos)

            # Update UI
            current_loop_ms = self.smooth_ms
            self.wave.set_playback_pos(current_loop_ms)
            
            if self.k_grn.value() > 0.05:
                self.wave.set_play_head(-1) 
            else:
                self.wave.set_play_head(current_loop_ms / total_dur_ms)
                
        else:
            self.last_media_pos = -1
            self.wave.set_play_head(0.0)
            self.wave.set_playback_pos(-1)

    def media_status(self, s):
        if s == QMediaPlayer.MediaStatus.EndOfMedia:
            # --- FIX START: Only stop if NOT granular ---
            # If granular, we let native setLoops(Infinite) handle it
            if self.k_grn.value() <= 0.05:
                self.wave.set_play_head(-1)
                self.atimer.stop()
                self.btn_play.set_state(2) # Paused/Ready

    def clear_state(self):
        self.player.stop()
        self.btn_play.set_state(0) # Idle
        self.wave.set_play_head(-1)
        self.atimer.stop()
        self.data, self.p_data = None, None
        self.dsp_cache, self.dsp_hash = None, None
        self.wave.set_data(None)
        self.player.setSource(QUrl())
    
    def toggle_playback(self):
        if self.p_data is None: return
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.set_state(2) # Paused
        else:
            self.player.play()
            self.btn_play.set_state(1) # Playing
    
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
