#include "webui.h"
#include "face_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <string>

namespace {

std::string http_resp(const char *status, const char *ctype, const std::string &body) {
    std::string s = "HTTP/1.0 ";
    s += status;
    s += "\r\nContent-Type: ";
    s += ctype;
    s += "\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: ";
    char len[32];
    snprintf(len, sizeof(len), "%u", (unsigned)body.size());
    s += len;
    s += "\r\n\r\n";
    s += body;
    return s;
}

std::string json_resp(const std::string &body) {
    return http_resp("200 OK", "application/json; charset=utf-8", body);
}

int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string url_decode(const std::string &s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '+') {
            o += ' ';
        } else if (s[i] == '%' && i + 2 < s.size() && hexval(s[i + 1]) >= 0 && hexval(s[i + 2]) >= 0) {
            o += (char)(hexval(s[i + 1]) * 16 + hexval(s[i + 2]));
            i += 2;
        } else {
            o += s[i];
        }
    }
    return o;
}

std::string form_field(const std::string &body, const char *key) {
    size_t pos = 0;
    while (pos <= body.size()) {
        size_t amp = body.find('&', pos);
        std::string kv = body.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
        size_t eq = kv.find('=');
        if (eq != std::string::npos) {
            std::string k = url_decode(kv.substr(0, eq));
            if (k == key) return url_decode(kv.substr(eq + 1));
        }
        if (amp == std::string::npos) break;
        pos = amp + 1;
    }
    return "";
}

const char *kPage = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Faces — LicheeRV Nano</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#e6edf3;font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{display:flex;gap:16px;padding:16px;min-height:100vh;align-items:flex-start;flex-wrap:wrap}
.video{flex:1 1 640px;min-width:320px;background:#000;border-radius:12px;overflow:hidden;border:1px solid #21262d}
.video img{display:block;width:100%;height:auto}
.side{flex:0 1 340px;min-width:280px;display:flex;flex-direction:column;gap:12px}
.card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:14px}
h1{font-size:18px;margin:0 0 2px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin:0 0 10px}
.people{display:flex;flex-wrap:wrap;gap:8px;min-height:30px}
.chip{display:inline-flex;align-items:center;gap:6px;background:#1f6feb22;border:1px solid #1f6feb66;color:#c9d1d9;border-radius:999px;padding:5px 6px 5px 12px;font-size:14px}
.chip button{background:none;border:0;color:#8b949e;cursor:pointer;padding:0 4px}
.chip button:hover{color:#f85149}
#dots{font-size:16px;line-height:1}
.empty{color:#6e7681;font-size:14px}
form{display:flex;gap:8px}
input{flex:1;background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:8px;padding:9px 11px;font:inherit;min-width:0}
input:focus{outline:none;border-color:#1f6feb}
button.primary{background:#238636;border:1px solid #2ea043;color:#fff;border-radius:8px;padding:9px 14px;font:inherit;cursor:pointer}
button.primary:hover{background:#2ea043}
button.primary:disabled{opacity:.5;cursor:default}
.status{margin-top:10px;font-size:14px;min-height:20px;color:#8b949e}
.status.wait{color:#d29922}
.status.ok{color:#3fb950}
.status.err{color:#f85149}
.hint{font-size:13px;color:#6e7681}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3fb950;margin-right:6px;vertical-align:middle}
@media(max-width:760px){.wrap{padding:10px;gap:10px}.side{flex:1 1 100%}}
</style>
</head>
<body>
<div class="wrap">
  <div class="video"><img src="/stream" alt="Video stream"></div>
  <aside class="side">
    <div class="card">
      <h1><span class="dot"></span>Face recognition</h1>
      <div class="hint">LicheeRV Nano · SCRFD + cviface</div>
    </div>
    <div class="card">
      <h2>People</h2>
      <div id="people" class="people"><span class="empty">loading…</span></div>
    </div>
    <div class="card">
      <h2>Liveness (automatic)</h2>
      <div id="live" class="status">waiting for a face</div>
      <div class="hint" style="margin-top:8px">The board tells a live person from a photo by itself: a person blinks (visible in the eyes), a photo never does. Nothing to do — just look at the camera for a couple of seconds.</div>
    </div>
    <div class="card">
      <h2>Remember a person</h2>
      <form id="f">
        <input id="name" placeholder="Name" maxlength="40" autocomplete="off" required>
        <button id="btn" class="primary" type="submit">Remember</button>
      </form>
      <div id="status" class="status"></div>
      <div class="hint" style="margin-top:8px">Stand in frame so the face is large and well lit; the board will remember several frames.</div>
    </div>
  </aside>
</div>
<script>
const q=s=>document.querySelector(s);
async function loadPeople(){
  const box=q('#people');
  try{
    const r=await fetch('/api/faces',{cache:'no-store'});
    const j=await r.json();
    box.innerHTML='';
    if(!j.people.length){box.innerHTML='<span class="empty">nobody yet</span>';return;}
    for(const name of j.people){
      const el=document.createElement('span');el.className='chip';
      const t=document.createElement('span');t.textContent=name;el.appendChild(t);
      const b=document.createElement('button');b.type='button';b.textContent='×';b.title='delete';
      b.onclick=async()=>{
        if(!confirm('Delete "'+name+'"?'))return;
        await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'name='+encodeURIComponent(name)});
        loadPeople();
      };
      el.appendChild(b);box.appendChild(el);
    }
  }catch(e){box.innerHTML='<span class="empty">no connection to the board</span>';}
}
async function poll(){
  const el=q('#status');
  try{
    const r=await fetch('/api/status',{cache:'no-store'});
    const j=await r.json();
    if(j.state==='collecting'){
      el.textContent='Look at the camera… '+j.progress+'%';el.className='status wait';
      setTimeout(poll,400);return;
    }
    if(j.state==='done'){el.textContent='Done: '+j.name;el.className='status ok';q('#btn').disabled=false;loadPeople();}
    else if(j.state==='error'){el.textContent=j.msg||'failed';el.className='status err';q('#btn').disabled=false;}
    else{el.textContent='';el.className='status';q('#btn').disabled=false;}
  }catch(e){el.textContent='no connection';el.className='status err';q('#btn').disabled=false;}
}
q('#f').addEventListener('submit',async ev=>{
  ev.preventDefault();
  const name=q('#name').value.trim();if(!name)return;
  const st=q('#status');q('#btn').disabled=true;
  try{
    const r=await fetch('/api/enroll',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'name='+encodeURIComponent(name)});
    const j=await r.json();
    if(!j.ok){st.textContent=j.error||'error';st.className='status err';q('#btn').disabled=false;return;}
    st.textContent='Look at the camera…';st.className='status wait';
    q('#name').value='';
    poll();
  }catch(e){st.textContent='no connection';st.className='status err';q('#btn').disabled=false;}
});
async function pollLive(){
  const el=q('#live');
  try{
    const r=await fetch('/api/liveness',{cache:'no-store'});
    const j=await r.json();
    if(j.state==='ok'){el.textContent=j.msg||'person';el.className='status ok';}
    else if(j.state==='photo'){el.textContent=j.msg||'looks like a photo';el.className='status err';}
    else{el.textContent=j.msg||'checking face…';el.className='status';}
  }catch(e){el.textContent='no connection';el.className='status err';}
  setTimeout(pollLive,700);
}
loadPeople();
pollLive();
</script>
</body>
</html>
)HTML";

} // namespace

std::string web_handle(const std::string &method, const std::string &path,
                       const std::string &body) {
    if (path == "/" || path == "/index.html") {
        return http_resp("200 OK", "text/html; charset=utf-8", kPage);
    }
    if (path == "/api/faces") {
        char buf[8192];
        face_gallery_json(buf, sizeof(buf));
        return json_resp(buf);
    }
    if (path == "/api/status") {
        char buf[1024];
        face_enroll_status(buf, sizeof(buf));
        return json_resp(buf);
    }
    if (path == "/api/enroll") {
        if (method != "POST") return http_resp("405 Method Not Allowed", "application/json", "{\"ok\":false}");
        std::string name = form_field(body, "name");
        if (name.empty()) return json_resp("{\"ok\":false,\"error\":\"empty name\"}");
        int rc = face_enroll_begin(name.c_str());
        if (rc == -1) return json_resp("{\"ok\":false,\"error\":\"invalid name\"}");
        if (rc == -2) return json_resp("{\"ok\":false,\"error\":\"registration already in progress\"}");
        if (rc == -3)
            return json_resp("{\"ok\":false,\"error\":\"wait a couple of seconds: face check in progress\"}");
        return json_resp("{\"ok\":true}");
    }
    if (path == "/api/liveness") {
        if (method == "POST") {
            face_liveness_reset();
            char buf[1024];
            face_liveness_json(buf, sizeof(buf));
            return json_resp(buf);
        }
        char buf[1024];
        face_liveness_json(buf, sizeof(buf));
        return json_resp(buf);
    }
    if (path == "/api/delete") {
        if (method != "POST") return http_resp("405 Method Not Allowed", "application/json", "{\"ok\":false}");
        std::string name = form_field(body, "name");
        if (name.empty()) return json_resp("{\"ok\":false,\"error\":\"empty name\"}");
        if (face_delete(name.c_str()) != 0) return json_resp("{\"ok\":false,\"error\":\"invalid name\"}");
        return json_resp("{\"ok\":true}");
    }
    return std::string();
}
