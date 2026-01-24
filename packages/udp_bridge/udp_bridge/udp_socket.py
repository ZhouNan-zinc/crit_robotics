import logging
import socket
import selectors
import threading
from functools import partial

class UdpSocket:
    BUFSIZE = 65535
    def __init__(self, local_ip, local_port, remote_ip: str, remote_port: int, *, timeout=1.0, logger = None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((local_ip, local_port))
        self.sock.connect((remote_ip, remote_port))
        self.sock.setblocking(False)

        self.timeout = timeout
        self.logger = logger or logging.getLogger("UdpSocket")

        self.on_message_callback = None
        self.on_error_callback = None

        self.executor = threading.Thread(target=self._recv_loop, daemon=True)
        self.spinlock = threading.Lock()
        self.running_event = threading.Event()

    def _recv_loop(self):
        sel = selectors.DefaultSelector()
        sel.register(self.sock, selectors.EVENT_READ)

        while self.running_event.is_set():
            if self.on_message_callback is None:
                self.logger.error("Callback not registered.")
                return

            events = sel.select(timeout=self.timeout)
            if not events:
                continue
            
            for key, _ in events:
                data, addr = key.fileobj.recvfrom(self.BUFSIZE)
                self.on_message_callback(data)

    def start(self) -> None:
        with self.spinlock:
            if not self.running_event.is_set():
                self.running_event.set()
                self.executor.start()
            else:
                return

    def stop(self) -> None:
        with self.spinlock:
            if self.running_event.is_set():
                self.running_event.clear()
                self.sock.close()
                self.executor.join()
            else:
                return

    def is_running(self) -> bool:
        return self.running_event.is_set()

    def on_message(self, callback) -> None:
        self.on_message_callback = callback

    def on_error(self, callback) -> None:
        self.on_error_callback = callback

    def send(self, data: bytes) -> int:
        try:
            return self.sock.sendto(data)
        except OSError as e:
            self.logger.error(f"UDP send failed: {e}")
            return 0

    def get_stats(self) -> dict: ...
    def reset_stats(self) -> None: ...
