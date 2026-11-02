"""A scripted OpenAI-compatible Chat Completions server on 127.0.0.1, for the end-to-end test.

``script(body) -> str`` returns the assistant text for each main-agent request (one that offers
tools). Other requests (session titles and similar side calls) get a fixed short answer.
Every request body is recorded in ``requests``.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeProvider:
    def __init__(self, script, model="fake-leaky-1"):
        self.script, self.model, self.requests = script, model, []
        self._lock = threading.Lock()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload, content_type="application/json"):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    return self._send(200, {"object": "list", "data": [
                        {"id": provider.model, "object": "model", "context_length": 131072}]})
                return self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": "not found"}})
                with provider._lock:
                    provider.requests.append(body)
                text = provider.script(body) if body.get("tools") else "Session"
                if body.get("stream"):
                    return self._send(200, provider._sse(text), "text/event-stream")
                return self._send(200, {
                    "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": provider.model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _sse(self, text):
        base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0, "model": self.model}
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {**base, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
        ]
        return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def main_requests(self):
        return [r for r in self.requests if r.get("tools")]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
