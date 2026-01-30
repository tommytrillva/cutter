#!/usr/bin/env python3
"""Simple web-based UI for Cutter video editor with native file dialogs."""

import http.server
import socketserver
import webbrowser
import json
import subprocess
import sys
import platform
from pathlib import Path
import yaml

PORT = 8765
PROJECT_DIR = Path(__file__).parent


def pick_file(title="Select a file", file_types=None):
    """Open native file picker dialog."""
    if platform.system() == "Darwin":
        # macOS - use AppleScript
        script = f'choose file with prompt "{title}"'
        if file_types:
            types = ", ".join(f'"{t}"' for t in file_types)
            script += f' of type {{{types}}}'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                # Convert "alias Macintosh HD:Users:..." to "/Users/..."
                alias = result.stdout.strip()
                if alias.startswith("alias "):
                    alias = alias[6:]
                # Split by : and rejoin with /
                parts = alias.split(":")
                if len(parts) > 1:
                    return "/" + "/".join(parts[1:])
            return None
        except:
            return None
    elif platform.system() == "Windows":
        # Windows - use PowerShell
        try:
            script = '''
            Add-Type -AssemblyName System.Windows.Forms
            $dialog = New-Object System.Windows.Forms.OpenFileDialog
            $dialog.Title = "''' + title + '''"
            if ($dialog.ShowDialog() -eq 'OK') { $dialog.FileName }
            '''
            result = subprocess.run(
                ["powershell", "-Command", script],
                capture_output=True, text=True, timeout=120
            )
            path = result.stdout.strip()
            return path if path else None
        except:
            return None
    return None


def pick_folder(title="Select a folder"):
    """Open native folder picker dialog."""
    if platform.system() == "Darwin":
        script = f'choose folder with prompt "{title}"'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                alias = result.stdout.strip()
                if alias.startswith("alias "):
                    alias = alias[6:]
                parts = alias.split(":")
                if len(parts) > 1:
                    return "/" + "/".join(parts[1:])
            return None
        except:
            return None
    elif platform.system() == "Windows":
        try:
            script = '''
            Add-Type -AssemblyName System.Windows.Forms
            $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
            $dialog.Description = "''' + title + '''"
            if ($dialog.ShowDialog() -eq 'OK') { $dialog.SelectedPath }
            '''
            result = subprocess.run(
                ["powershell", "-Command", script],
                capture_output=True, text=True, timeout=120
            )
            path = result.stdout.strip()
            return path if path else None
        except:
            return None
    return None


HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Cutter - Video Editor</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 700px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 {
            text-align: center;
            color: #fff;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #aaa;
        }
        .file-input {
            display: flex;
            gap: 10px;
        }
        input[type="text"], select {
            width: 100%;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #16213e;
            color: #fff;
            font-size: 14px;
        }
        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #e94560;
        }
        .browse-btn {
            padding: 12px 20px;
            background: #0f3460;
            color: white;
            border: 1px solid #333;
            border-radius: 8px;
            cursor: pointer;
            white-space: nowrap;
            font-size: 14px;
        }
        .browse-btn:hover {
            background: #1a4a7a;
        }
        .row {
            display: flex;
            gap: 20px;
        }
        .row .form-group {
            flex: 1;
        }
        button.primary {
            width: 100%;
            padding: 15px;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
        }
        button.primary:hover {
            background: #ff6b6b;
        }
        button:disabled {
            background: #555 !important;
            cursor: not-allowed;
        }
        #status {
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }
        #status.running {
            display: block;
            background: #16213e;
            color: #4ecca3;
        }
        #status.success {
            display: block;
            background: #1b4332;
            color: #95d5b2;
        }
        #status.error {
            display: block;
            background: #3d1515;
            color: #f8d7da;
        }
        .progress {
            width: 100%;
            height: 6px;
            background: #333;
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: #e94560;
            width: 0%;
            transition: width 0.3s;
        }
        hr {
            border: none;
            border-top: 1px solid #333;
            margin: 25px 0;
        }
    </style>
</head>
<body>
    <h1>Cutter</h1>

    <div class="form-group">
        <label>Video Folder</label>
        <div class="file-input">
            <input type="text" id="video" placeholder="Click Browse to select folder...">
            <button class="browse-btn" onclick="browseFolder('video')">Browse</button>
        </div>
    </div>

    <div class="form-group">
        <label>Music File</label>
        <div class="file-input">
            <input type="text" id="audio" placeholder="Click Browse to select...">
            <button class="browse-btn" onclick="browseFile('audio')">Browse</button>
        </div>
    </div>

    <div class="form-group">
        <label>Output Folder</label>
        <div class="file-input">
            <input type="text" id="output" value="./output">
            <button class="browse-btn" onclick="browseFolder('output')">Browse</button>
        </div>
    </div>

    <hr>

    <div class="row">
        <div class="form-group">
            <label>Duration (seconds)</label>
            <input type="text" id="duration" value="60">
        </div>
        <div class="form-group">
            <label>Platform</label>
            <select id="platform">
                <option value="tiktok">TikTok</option>
                <option value="instagram_reels">Instagram Reels</option>
                <option value="youtube_shorts">YouTube Shorts</option>
                <option value="youtube">YouTube</option>
            </select>
        </div>
    </div>

    <div class="row">
        <div class="form-group">
            <label>Variations</label>
            <select id="variations">
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3" selected>3</option>
                <option value="4">4</option>
                <option value="5">5</option>
            </select>
        </div>
        <div class="form-group">
            <label>Color Style</label>
            <select id="color">
                <option value="cinematic">Cinematic</option>
                <option value="cyberpunk">Cyberpunk</option>
                <option value="vintage">Vintage</option>
                <option value="VHS">VHS</option>
                <option value="neon">Neon</option>
                <option value="noir">Noir</option>
                <option value="warm">Warm</option>
                <option value="cool">Cool</option>
            </select>
        </div>
    </div>

    <button class="primary" id="generate" onclick="generate()">Generate Video</button>

    <div id="status">
        <div id="statusText"></div>
        <div class="progress"><div class="progress-bar" id="progressBar"></div></div>
    </div>

    <script>
        async function browseFile(inputId) {
            const type = inputId === 'video' ? 'video' : 'audio';
            try {
                const response = await fetch('/browse', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type: type, mode: 'file'})
                });
                const result = await response.json();
                if (result.path) {
                    document.getElementById(inputId).value = result.path;
                }
            } catch (e) {
                console.error('Browse error:', e);
            }
        }

        async function browseFolder(inputId) {
            const title = inputId === 'video' ? 'Select Video Folder' : 'Select Output Folder';
            try {
                const response = await fetch('/browse', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type: inputId, mode: 'folder', title: title})
                });
                const result = await response.json();
                if (result.path) {
                    document.getElementById(inputId).value = result.path;
                }
            } catch (e) {
                console.error('Browse error:', e);
            }
        }

        async function generate() {
            const video = document.getElementById('video').value.trim();
            const audio = document.getElementById('audio').value.trim();
            const output = document.getElementById('output').value.trim();
            const duration = document.getElementById('duration').value.trim();
            const platform = document.getElementById('platform').value;
            const variations = document.getElementById('variations').value;
            const color = document.getElementById('color').value;

            if (!video) { alert('Please select a video file'); return; }
            if (!audio) { alert('Please select a music file'); return; }

            const btn = document.getElementById('generate');
            const status = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            const progressBar = document.getElementById('progressBar');

            btn.disabled = true;
            status.className = 'running';
            statusText.textContent = 'Processing...';
            progressBar.style.width = '10%';

            try {
                const response = await fetch('/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({video, audio, output, duration, platform, variations, color})
                });

                const result = await response.json();

                if (result.success) {
                    status.className = 'success';
                    statusText.textContent = 'Done! Video saved to: ' + output;
                    progressBar.style.width = '100%';
                } else {
                    status.className = 'error';
                    statusText.textContent = 'Error: ' + result.error;
                }
            } catch (e) {
                status.className = 'error';
                statusText.textContent = 'Error: ' + e.message;
            }

            btn.disabled = false;
        }
    </script>
</body>
</html>
"""


class CutterHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())

        if self.path == '/browse':
            result = self.handle_browse(data)
        elif self.path == '/run':
            result = self.run_pipeline(data)
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def handle_browse(self, data):
        mode = data.get('mode', 'file')
        file_type = data.get('type', 'video')
        title = data.get('title', 'Select')

        if mode == 'folder':
            path = pick_folder(title)
        else:
            if file_type == 'video':
                path = pick_file("Select Video File", ["mp4", "mov", "avi", "mkv", "m4v"])
            else:
                path = pick_file("Select Music File", ["mp3", "wav", "m4a", "aac", "flac"])

        return {'path': path}

    def run_pipeline(self, data):
        try:
            video_folder = Path(data['video'])
            audio_path = Path(data['audio'])

            if not video_folder.exists():
                return {'success': False, 'error': f'Video folder not found: {data["video"]}'}
            if not video_folder.is_dir():
                return {'success': False, 'error': f'Video path must be a folder: {data["video"]}'}
            if not audio_path.exists():
                return {'success': False, 'error': f'Audio file not found: {data["audio"]}'}

            variations_count = int(data['variations'])
            config = {
                'project_name': video_folder.name,
                'input_video_folder': str(video_folder),
                'audio_track': str(audio_path),
                'output_directory': data['output'],
                'audio_analysis': {
                    'audio_sensitivity': 'medium',
                    'beat_detection_threshold': 0.5
                },
                'shot_quality': {
                    'motion_weight': 0.25,
                    'face_detection_weight': 0.25,
                    'composition_weight': 0.15,
                    'contrast_weight': 0.25,
                    'lighting_weight': 0.10,
                    'min_quality_score': 0.4
                },
                'cut_timing': {
                    'target_video_length': int(data['duration']),
                    'min_cut_duration': 0.5,
                    'max_cut_duration': 5.0,
                    'cut_pattern': 'balanced'
                },
                'motion_graphics': {
                    'enabled': True,
                    'intensity': 'high',
                    'text': {'enabled': False},
                    'transitions': {'enabled': True, 'style': 'dynamic', 'duration': 0.3},
                    'particles': {'enabled': True, 'type': 'confetti', 'density': 'medium', 'audio_reactive': True}
                },
                'color_grading': {
                    'enabled': True,
                    'preset_profile': data['color'],
                    'saturation': 1.0,
                    'contrast': 1.0,
                    'brightness': 1.0
                },
                'export': {
                    'formats': ['mp4'],
                    'capcut_projects': True,
                    'fcp_xml': False,
                    'video': {'codec': 'h264', 'quality': 18, 'fps': 30},
                    'audio': {'codec': 'aac', 'bitrate': '192k'},
                    'platform': data['platform']
                },
                'variations': {
                    'count': variations_count,
                    'sensitivities': ['low', 'medium', 'high'][:variations_count]
                }
            }

            config_path = PROJECT_DIR / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            cmd = [sys.executable, '-m', 'src.main', '--config', str(config_path)]
            result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)

            if result.returncode == 0:
                return {'success': True}
            else:
                return {'success': False, 'error': result.stderr or result.stdout or 'Unknown error'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def log_message(self, format, *args):
        pass


def main():
    # Allow socket reuse
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", PORT), CutterHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"Cutter UI running at {url}")
        print("Press Ctrl+C to stop")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
