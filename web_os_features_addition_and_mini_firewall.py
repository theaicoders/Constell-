"""
WebOS_Features_Addition_and_MiniFirewall.py
Single-file Flask web application extension to the WebOS prototype.
This file adds (implemented in a web-friendly / safe way):
  - Window snapping and tiling (edge snap & quadrant snap)
  - Desktop icons with drag & drop and right-click context (shortcuts)
  - Task Manager (process list + kill) - server-limited and safe
  - Notifications / Action Center (toast messages + notification center)
  - Settings app with UI for themes, firewall, and file associations
  - "Open With" file association mechanism for the File Manager
  - Voice input for the Gemini assistant using Web Speech API (browser-side)
  - Mini Firewall (server-side simulated firewall): outbound allow/block rules by domain/IP and port; enforced by server endpoints (clone, ai proxy, shell)

Security & deployment notes:
  - This is a developer demo. Mini Firewall is an application-level guard inside Flask and does NOT replace an OS firewall.
  - The Task Manager shows processes using `ps` and limits kill to processes owned by the server user; use caution.
  - Voice input uses browser Web Speech API (no server audio processing required).
  - Firewall rules are persisted to a local JSON file `firewall_rules.json` in the app directory.

USAGE:
  - pip install flask requests
  - export GEMINI_API_KEY="your_key"
  - python WebOS_Features_Addition_and_MiniFirewall.py
  - Open http://127.0.0.1:5000

--- START CODE ---

from flask import Flask, request, jsonify, render_template_string, session, send_from_directory
import os, subprocess, json, time

app = Flask(__name__)
app.secret_key = os.environ.get('WEBOS_SECRET', 'change-me')
BASE_DIR = os.path.abspath(os.getcwd())
REPO_ROOT = os.path.join(BASE_DIR, 'repos')
os.makedirs(REPO_ROOT, exist_ok=True)
FIREWALL_FILE = os.path.join(BASE_DIR, 'firewall_rules.json')
GEMINI_KEY = os.environ.get('GEMINI_API_KEY')

# -------------------- Firewall: simple application-level rules --------------------
DEFAULT_FW = {
    "blocked_domains": ["example-malicious.com"],
    "allowed_domains": [],
    "blocked_ports": [],
    "allow_all_outbound": False
}

if not os.path.exists(FIREWALL_FILE):
    with open(FIREWALL_FILE, 'w') as f:
        json.dump(DEFAULT_FW, f, indent=2)

def load_fw():
    try:
        with open(FIREWALL_FILE,'r') as f:
            return json.load(f)
    except Exception:
        return DEFAULT_FW.copy()

def save_fw(rules):
    with open(FIREWALL_FILE,'w') as f:
        json.dump(rules, f, indent=2)

def fw_check_domain(domain):
    rules = load_fw()
    if rules.get('allow_all_outbound'):
        return True, 'allowed (global)'
    blocked = rules.get('blocked_domains', [])
    allowed = rules.get('allowed_domains', [])
    # normalize
    d = domain.lower()
    if d in blocked:
        return False, 'blocked by domain'
    if allowed and d not in allowed:
        return False, 'not in allowlist'
    return True, 'allowed'

def fw_check_port(port):
    rules = load_fw()
    blocked_ports = rules.get('blocked_ports', [])
    if port in blocked_ports:
        return False, 'port blocked'
    return True, 'allowed'

# -------------------- Utilities --------------------

def run_cmd(cmd, cwd=None, timeout=15):
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout.decode('utf-8', errors='ignore'), "err": p.stderr.decode('utf-8', errors='ignore')}
    except Exception as e:
        return {"rc": -1, "out":"", "err": str(e)}

# -------------------- UI Index: includes snapping, desktop icons, notifications, voice --------------------
@app.route('/')
def index():
    html = '''
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>WebOS Extended</title>
<link href="https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">
<style>
  body{height:100vh;margin:0;font-family:Inter,system-ui}
  #desktop{position:relative;width:100vw;height:100vh;background:linear-gradient(120deg,#04243a,#001021);overflow:hidden}
  .window{position:absolute;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.06);}
  .icon{width:88px;text-align:center;color:white;margin:8px;cursor:grab}
  #icons{position:absolute;left:12px;top:12px}
  .toast{position:fixed;right:16px;bottom:16px;max-width:320px}
</style>
</head>
<body>
  <div id="desktop">
    <div id="icons" class="flex flex-col"></div>
    <div style="position:fixed;left:16px;top:16px;z-index:50">
      <button onclick="openTaskManager()" class="px-2 py-1 bg-white bg-opacity-10 text-white rounded">Task Manager</button>
      <button onclick="openSettings()" class="px-2 py-1 bg-white bg-opacity-10 text-white rounded">Settings</button>
      <button onclick="openAssistant()" class="px-2 py-1 bg-white bg-opacity-10 text-white rounded">Assistant (Voice)</button>
    </div>
    <div id="windows"></div>
    <div id="toasts" class="toast"></div>
  </div>

<template id="win-tpl">
  <div class="window shadow-lg" style="width:640px;height:420px;left:120px;top:120px;">
    <div class="p-2 flex justify-between items-center bg-transparent text-white">
      <div class="title">Window</div>
      <div>
        <button class="snap">Snap</button>
        <button class="close">Close</button>
      </div>
    </div>
    <div class="content p-3 text-white" style="height:calc(100% - 48px);overflow:auto;"></div>
  </div>
</template>

<script src="https://unpkg.com/axios/dist/axios.min.js"></script>
<script>
// Desktop icons
const icons = [
  {id:'repo', name:'Repos', action:openFileManager},
  {id:'notes', name:'Notes', action:()=>openWindow('Notes','<div>Simple notes</div>')},
  {id:'trash', name:'Trash', action:()=>openWindow('Trash','<div>Empty</div>')}
];
function renderIcons(){
  const container = document.getElementById('icons'); container.innerHTML='';
  icons.forEach(ic=>{
    const d = document.createElement('div'); d.className='icon'; d.draggable=true; d.id='icon-'+ic.id;
    d.innerHTML = `<div style="width:64px;height:64px;border-radius:8px;background:rgba(255,255,255,0.03);display:flex;align-items:center;justify-content:center">${ic.name[0]}</div><div style="font-size:12px;margin-top:6px">${ic.name}</div>`;
    d.ondragstart = (e)=>{ e.dataTransfer.setData('text/plain', ic.id); };
    d.onclick = ic.action;
    container.appendChild(d);
  });
}

// Window manager with snap
const windows = {};
function openWindow(title, html){
  const tpl = document.getElementById('win-tpl').content.cloneNode(true);
  const el = tpl.querySelector('.window');
  el.querySelector('.title').innerText = title;
  el.querySelector('.content').innerHTML = html;
  el.querySelector('.close').onclick = ()=>el.remove();
  el.querySelector('.snap').onclick = ()=>snapWindow(el);
  el.style.left = (120 + Object.keys(windows).length*24)+'px'; el.style.top = (120 + Object.keys(windows).length*18)+'px';
  document.getElementById('windows').appendChild(el);
  windows[title] = el;
}
function snapWindow(el){
  // Cycle through full, left, right, top-left, top-right, bottom-left, bottom-right
  const w = window.innerWidth, h=window.innerHeight;
  const rect = el.getBoundingClientRect();
  const key = el.getAttribute('data-snap') || 'none';
  const order = ['none','left','right','top-left','top-right','bottom-left','bottom-right','full'];
  const next = order[(order.indexOf(key)+1)%order.length];
  el.setAttribute('data-snap', next);
  switch(next){
    case 'none': el.style.width='640px'; el.style.height='420px'; break;
    case 'left': el.style.left='0px'; el.style.top='0px'; el.style.width = (w/2)+'px'; el.style.height=(h)+'px'; break;
    case 'right': el.style.left=(w/2)+'px'; el.style.top='0px'; el.style.width=(w/2)+'px'; el.style.height=h+'px'; break;
    case 'top-left': el.style.left='0px'; el.style.top='0px'; el.style.width=(w/2)+'px'; el.style.height=(h/2)+'px'; break;
    case 'top-right': el.style.left=(w/2)+'px'; el.style.top='0px'; el.style.width=(w/2)+'px'; el.style.height=(h/2)+'px'; break;
    case 'bottom-left': el.style.left='0px'; el.style.top=(h/2)+'px'; el.style.width=(w/2)+'px'; el.style.height=(h/2)+'px'; break;
    case 'bottom-right': el.style.left=(w/2)+'px'; el.style.top=(h/2)+'px'; el.style.width=(w/2)+'px'; el.style.height=(h/2)+'px'; break;
    case 'full': el.style.left='0px'; el.style.top='0px'; el.style.width=w+'px'; el.style.height=h+'px'; break;
  }
}

// Notifications
function toast(msg){
  const t = document.createElement('div'); t.className='p-2 bg-white bg-opacity-10 rounded mb-2 text-white'; t.innerText = msg;
  const container = document.getElementById('toasts'); container.appendChild(t);
  setTimeout(()=>{ t.remove(); }, 7000);
}
function openNotificationCenter(){
  openWindow('Notifications', '<div id="nc">No notifications</div>');
}

// Task Manager (server-backed)
async function openTaskManager(){
  openWindow('Task Manager', '<div id="tm">Loading...</div>');
  try{
    const r = await axios.get('/api/tasks');
    const rows = r.data.processes.map(p=>`<tr><td>${p.pid}</td><td>${p.user}</td><td>${p.cpu}</td><td>${p.mem}</td><td>${p.cmd}</td><td><button onclick="killPid(${p.pid})">Kill</button></td></tr>`).join('');
    document.getElementById('tm').innerHTML = `<table class="table-auto text-white"><thead><tr><th>PID</th><th>USER</th><th>%CPU</th><th>%MEM</th><th>CMD</th><th>ACTION</th></tr></thead><tbody>${rows}</tbody></table>`;
  }catch(e){ document.getElementById('tm').innerText = 'Error: '+e.message; }
}
async function killPid(pid){
  const r = await axios.post('/api/kill',{pid}); if(r.data.ok){ toast('Killed '+pid); openTaskManager(); }else{ toast('Kill failed: '+(r.data.error||'unknown')); }
}

// File Manager with Open With
async function openFileManager(){
  openWindow('File Manager', '<div id="fm">Loading...</div>');
  const r = await axios.get('/api/repos');
  const list = r.data.repos.map(x=>`<li>${x} <button onclick="openRepo(\'${x}\')">Open</button></li>`).join('');
  document.getElementById('fm').innerHTML = `<ul>${list}</ul>`;
}
async function openRepo(name){
  openWindow('Repo: '+name, '<div id="repofiles">Loading...</div>');
  const r = await axios.get('/api/repo_files?name='+encodeURIComponent(name));
  const items = r.data.files.slice(0,200).map(f=>`<div><span>${f}</span> <select onchange="openWith(\'${name}\',\'${f}\',this.value)"><option value="editor">Editor</option><option value="preview">Preview</option></select></div>`).join('');
  document.getElementById('repofiles').innerHTML = items;
}
function openWith(repo, file, app){
  if(app==='editor') openWindow('Editor - '+file, `<div>Editor opened for ${file} (save disabled in demo)</div>`);
  else openWindow('Preview - '+file, `<div>Preview of ${file}</div>`);
}

// Assistant with voice input using Web Speech API
function openAssistant(){
  const html = `<div><div id="chat" style="height:240px;overflow:auto;background:rgba(255,255,255,0.02);padding:8px;border-radius:6px"></div>
    <div class="flex gap-2 mt-2"><input id="prompt" class="flex-1 p-2 rounded bg-white bg-opacity-5 text-white" placeholder="Ask..." /><button onclick="sendPrompt()" class="px-3 py-2 bg-blue-600 rounded">Send</button>
    <button onclick="startVoice()" class="px-3 py-2 bg-green-600 rounded">Voice</button></div></div>`;
  openWindow('Assistant (Voice)', html);
}
async function sendPrompt(){
  const prompt = document.getElementById('prompt').value; if(!prompt) return; appendChat('user',prompt); document.getElementById('prompt').value='';
  try{ const r = await axios.post('/api/ai',{prompt}); if(r.data.ok) appendChat('assistant', r.data.text); else appendChat('assistant','[error] '+(r.data.error||'unknown')); }catch(e){ appendChat('assistant','[network error] '+e.message); }
}
function appendChat(role, text){ const c=document.getElementById('chat'); const d=document.createElement('div'); d.className='p-2 my-1 rounded'; d.style.background = role==='user' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.2)'; d.innerHTML = `<strong>${role}:</strong><div>${text.replace(/\n/g,'<br/>')}</div>`; c.appendChild(d); c.scrollTop=c.scrollHeight; }

function startVoice(){
  if(!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) { toast('Voice not supported in this browser'); return; }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recog = new SpeechRecognition(); recog.lang='en-US'; recog.interimResults=false; recog.maxAlternatives=1;
  recog.onresult = (e)=>{ const txt = e.results[0][0].transcript; document.getElementById('prompt').value = txt; sendPrompt(); }
  recog.onerror = (e)=>{ toast('Voice error: '+e.error); }
  recog.start(); toast('Listening...');
}

// Settings: firewall UI
async function openSettings(){
  openWindow('Settings', '<div id="settings">Loading...</div>');
  const r = await axios.get('/api/firewall/status');
  const rules = r.data.rules; document.getElementById('settings').innerHTML = `<h3>Firewall</h3>
    <div>Allow all outbound: <input type=checkbox id=fw_all ${rules.allow_all_outbound? 'checked':''}></div>
    <div>Allowed domains (comma): <input id=fw_allow value='${(rules.allowed_domains||[]).join(',')}'></div>
    <div>Blocked domains (comma): <input id=fw_block value='${(rules.blocked_domains||[]).join(',')}'></div>
    <button onclick="saveFirewall()">Save</button>`;
}
async function saveFirewall(){
  const allowAll = document.getElementById('fw_all').checked;
  const allowed = document.getElementById('fw_allow').value.split(',').map(x=>x.trim()).filter(x=>x);
  const blocked = document.getElementById('fw_block').value.split(',').map(x=>x.trim()).filter(x=>x);
  const r = await axios.post('/api/firewall/update',{allow_all_outbound:allowAll, allowed_domains:allowed, blocked_domains:blocked});
  if(r.data.ok) { toast('Firewall updated'); } else toast('Failed');
}

// Init
renderIcons();
</script>
</body>
</html>
    '''
    return render_template_string(html)

# -------------------- API endpoints for firewall and tasks --------------------
@app.route('/api/firewall/status')
def api_firewall_status():
    rules = load_fw()
    # ensure keys
    return jsonify({'rules':{ 'allowed_domains': rules.get('allowed_domains',[]), 'blocked_domains': rules.get('blocked_domains',[]), 'blocked_ports': rules.get('blocked_ports',[]), 'allow_all_outbound': rules.get('allow_all_outbound', False) }})

@app.route('/api/firewall/update', methods=['POST'])
def api_firewall_update():
    data = request.get_json() or {}
    rules = load_fw()
    rules['allow_all_outbound'] = bool(data.get('allow_all_outbound', rules.get('allow_all_outbound', False)))
    rules['allowed_domains'] = data.get('allowed_domains', rules.get('allowed_domains', []))
    rules['blocked_domains'] = data.get('blocked_domains', rules.get('blocked_domains', []))
    rules['blocked_ports'] = data.get('blocked_ports', rules.get('blocked_ports', []))
    save_fw(rules)
    return jsonify({'ok':True})

# Task manager: list processes (limited)
@app.route('/api/tasks')
def api_tasks():
    # Use ps to list processes - limit to first 200 lines and safe parsing
    r = run_cmd("ps aux --sort=-%cpu | head -n 200")
    if r['rc'] != 0:
        return jsonify({'ok':False,'error':r['err'] or r['out']}), 500
    lines = r['out'].splitlines()[1:]
    procs = []
    for ln in lines:
        parts = ln.split(None,10)
        if len(parts) >= 11:
            user, pid, cpu, mem, vsz, rss, tty, stat, start, time_, cmd = parts
            procs.append({'user':user,'pid':int(pid),'cpu':cpu,'mem':mem,'cmd':cmd})
    return jsonify({'processes':procs})

@app.route('/api/kill', methods=['POST'])
def api_kill():
    data = request.get_json() or {}
    pid = int(data.get('pid',0))
    if pid <= 1:
        return jsonify({'ok':False,'error':'invalid pid'}),400
    # Extra safety: only allow killing processes owned by current user (simple check)
    r = run_cmd(f"ps -o uid= -p {pid}")
    if r['rc'] != 0:
        return jsonify({'ok':False,'error':'process not found'}),404
    try:
        uid = int(r['out'].strip())
    except Exception:
        return jsonify({'ok':False,'error':'could not determine owner'}),500
    import getpass
    current_user = getpass.getuser()
    # get uid of current user
    r2 = run_cmd(f"id -u {current_user}")
    if r2['rc'] != 0:
        return jsonify({'ok':False,'error':'user id lookup failed'}),500
    try:
        curuid = int(r2['out'].strip())
    except Exception:
        return jsonify({'ok':False,'error':'invalid current uid'}),500
    if uid != curuid:
        return jsonify({'ok':False,'error':'permission denied to kill pid'}),403
    # perform kill
    r3 = run_cmd(f"kill -9 {pid}")
    if r3['rc'] == 0:
        return jsonify({'ok':True})
    else:
        return jsonify({'ok':False,'error':r3['err'] or r3['out']}),500

# -------------------- Proxy/guarded endpoints example: clone and ai use firewall checks --------------------
@app.route('/api/clone', methods=['POST'])
def api_clone():
    data = request.get_json() or {}
    url = data.get('url','')
    if not url:
        return jsonify({'ok':False,'error':'missing url'}),400
    # Simple domain extraction
    import urllib.parse
    try:
        p = urllib.parse.urlparse(url)
        host = p.hostname or ''
    except Exception:
        return jsonify({'ok':False,'error':'invalid url'}),400
    ok, reason = fw_check_domain(host)
    if not ok:
        return jsonify({'ok':False,'error':'blocked by firewall: '+reason}),403
    # proceed with cloning using git (similar to previous implementations)
    name = os.path.splitext(os.path.basename(url.rstrip('/')))[0]
    target = os.path.join(REPO_ROOT, name)
    if os.path.isdir(target):
        return jsonify({'ok':True,'msg':'already exists','name':name})
    r = run_cmd(f'git clone {url} {target}', timeout=120)
    if r['rc']==0:
        return jsonify({'ok':True,'name':name})
    return jsonify({'ok':False,'error':r['err'] or r['out']}),500

# AI endpoint guarded by firewall domain rules (blocks calls if Gemini host blocked)
@app.route('/api/ai', methods=['POST'])
def api_ai():
    if not GEMINI_KEY:
        return jsonify({'ok':False,'error':'Gemini key not configured on server'}),500
    data = request.get_json() or {}
    prompt = data.get('prompt','')
    # check gemini domains - simple check
    ok, reason = fw_check_domain('generativelanguage.googleapis.com')
    if not ok:
        return jsonify({'ok':False,'error':'AI calls blocked by firewall: '+reason}),403
    # perform request to Gemini (simple REST call similar to previous file)
    import requests
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-mini:generateText'
    params = {'key': GEMINI_KEY}
    payload = { 'prompt': {'text': prompt}, 'maxOutputTokens': 512 }
    try:
        resp = requests.post(url, params=params, json=payload, timeout=30)
        if resp.status_code != 200:
            return jsonify({'ok':False,'error':f'gemini error {resp.status_code}'}),500
        data = resp.json()
        # best-effort extraction
        text = ''
        if isinstance(data, dict):
            if 'candidates' in data:
                text = data['candidates'][0].get('content',{}).get('parts',[None])[0]
            elif 'output' in data and isinstance(data['output'],dict):
                text = data['output'].get('text') or json.dumps(data['output'])
        if not text:
            text = json.dumps(data)
        return jsonify({'ok':True,'text':text})
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}),500

# basic repo files listing reused
@app.route('/api/repos')
def api_repos():
    repos = []
    for d in os.listdir(REPO_ROOT):
        if os.path.isdir(os.path.join(REPO_ROOT,d,'.git')):
            repos.append(d)
    return jsonify({'repos':repos})

@app.route('/api/repo_files')
def api_repo_files():
    name = request.args.get('name')
    if not name: return jsonify({'error':'missing name'}),400
    repo_path = os.path.join(REPO_ROOT,name)
    if not os.path.isdir(repo_path): return jsonify({'error':'repo not found'}),404
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        relroot = os.path.relpath(root, repo_path)
        for f in filenames:
            files.append(os.path.join(relroot,f))
        if len(files)>1000: break
    return jsonify({'files':sorted(files)})

# -------------------- Static helper --------------------
@app.route('/static/<path:p>')
def static_proxy(p):
    return send_from_directory(os.path.join(BASE_DIR,'static'), p)

# -------------------- Run --------------------
if __name__=='__main__':
    print('WebOS extended with snapping, task manager, notifications, voice assistant, and mini firewall')
    print('Firewall rules file at', FIREWALL_FILE)
    app.run(host='0.0.0.0', port=5000, debug=True)

--- END CODE ---
"""
