# import rclpy

# import os
# import socket
# import logging
# import threading
# import queue
# import yaml
# import struct
# from typing import Callable
# from rclpy.publisher import Publisher
# from rclpy.node import Node
# from ament_index_python import get_package_share_directory
# from std_msgs.msg import String
# from sensor_msgs.msg import Imu
# from std_msgs.msg import Float32MultiArray, Float32
# import rosidl_parser
# from rosidl_parser.definition import (
#     BasicType,
#     NamespacedType,
#     AbstractNestedType,
# )
# import rosidl_runtime_py
# from threading import Thread


# class UdpProtocolNodeInterface(Node):
#     node_name = "udpsocket_python"
#     def __init__(self, node_name, *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, rosout_qos_profile = ..., start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False, enable_logger_service = False):
#         super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, rosout_qos_profile=rosout_qos_profile, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides, enable_logger_service=enable_logger_service)

#         self.logger = self.get_logger()

#         self.sock = UdpSocket(
#             ip_addr=self.get_parameter_or("ip_addr", "127.0.0.1"),
#             port=self.get_parameter_or("port", "6060"),
#             callback=self.parse_datapack,
#             logger=self.logger
#         )

#         self.magic_number = self.get_parameter_or("magic_number", 0xA55A)
#         self.version = self.get_parameter_or("version", 1)
#         # type

#         self.executor = Thread()

#         ip_addr = self.get_parameter_or("ip_addr", "127.0.0.1")
#         port = self.get_parameter_or("port", 7000)



#         url = self.get_parameter_or("url", f"package://{self.node_name}/config/channels.yaml")

#         if url.startswith("package://"):
#             package_and_path = url[len("package://"):]
#             package, _, rel_path = package_and_path.partition("/")
#             if package:
#                 base_dir = get_package_share_directory(package)
#                 url = os.path.join(base_dir, rel_path) if rel_path else base_dir

#         with open(url, "r", encoding="utf-8") as f:
#             channels = yaml.safe_load(f)
        
#         for ch in channels:
#             if ch["direction"] == "publisher":
#                 self.create_publisher(
#                     ch["type"],
#                     ch["topic"],
#                     ch["qos"]
#                 )
#             elif ch["direction"] == "subscription":
#                 self.create_subscription(
#                     ch["type"],
#                     ch["topic"],
#                     ch["qos"]
#                     callback
#                 )
#             else:
#                 raise ValueError("key `direction` must be `publisher` or `subscription`")
    
#     def generate_message(self, data):
#         from rosidl_runtime_py.utilities import get_message
#         publisher = self.get_publisher(topic_name)
        
#         msg = self._generate_message_impl(data, publisher.msg_type)

    
#     def _generate_message_impl(self, data:list|tuple, msg_type:type):
#         if not isinstance(data, list):
#             data = list(data)

#         data.reverse()

#         msg:Imu = msg_type()

#         for slot, slot_type in zip(msg.get_fields_and_field_types().keys(), msg.SLOT_TYPES):
#             if isinstance(slot_type, BasicType):
#                 setattr(msg, slot, data.pop())
#             elif isinstance(slot_type, NamespacedType):
#                 setattr(msg, slot, self._generate_message_impl(data, slot_type))
#             elif isinstance(slot_type, AbstractNestedType):
#                 setattr(msg, slot, [self._generate_message_impl(data, slot_type)] for _ in len(slo))
#                 for submsg in msg:
#                     subfmt, subcontent = self._generate_message_impl(submsg)
#                     fmt += subfmt
#                     content.extend(subcontent)
        
#         return msg

#     def generate_datapack(self, msg:Imu):
#         data = []
#         for m in [getattr(msg, field) for field in msg.get_fields_and_field_types().keys()]:
#             if isinstance(m, BasicType):
#                 data.append(m)
#             else:
#                 data.extend()
#         return data

#     def _generate_datapack_impl(self, msg):
#         fmt = ""
#         data = []

#         for slot, slot_type in zip(msg.get_fields_and_field_types().keys(), msg.SLOT_TYPES):
#             if isinstance(slot_type, BasicType):
#                 value = getattr(msg, slot)
#                 content.append(value)
#                 if isinstance(value, int):
#                     fmt += "I"
#                 elif isinstance(value, float):
#                     fmt += "d"
#                 elif isinstance(value, str):
#                     raise ValueError("String is not supported.")
#                 else:
#                     raise ValueError(f"Unsupported slot type {type(value)}")
#             elif isinstance(slot_type, NamespacedType):
#                 submsg = getattr(msg, slot)
#                 subfmt, subcontent = self._generate_message_impl(submsg)
#                 fmt += subfmt
#                 content.extend(subcontent)
#             elif isinstance(slot_type, AbstractNestedType):
#                 for submsg in msg:
#                     subfmt, subcontent = self._generate_message_impl(submsg)
#                     fmt += subfmt
#                     content.extend(subcontent)

#         return fmt, content


#     def generate_message(self, data):
#         header_format = self.get_parameter_or("header_format", "HBBH")
#         header_size = struct.calcsize(header_format)

#         if len(data) < header_size:
#             self.logger.error("Got unexpected EOF while parsing datapack.")
#             return
            
#         magic_number, version, sequence_number, timestamp, message_id = struct.unpack_from(
#             header_format,
#             buffer=data,
#             offset=0
#         )

#         if version != self.version:
#             self.logger.error("Version mismatch.")
#             return

#         if magic_number != self.magic_number:
#             self.logger.error("Datapack transfer error: magic number mismatch.")
#             return
        
#         data = struct.unpack_from(
#             self.get_message_format(self.get_topic_name(message_id)),
#             buffer=data,
#             offset=header_size
#         )

#         # TODO: check sequence number, calculate loss pack rate

#         # TODO: check timestamp, calculate transfer delay

#         self.generate_message(data, self.get_topic_name(message_id))


#     def get_topic_name(self, message_id):
#         pass

#     def get_publisher(self, topic_name) -> Publisher:
#         pass

#     def get_message_format(self, topic_name:str):
#         message = 

# class UdpSocket:
#     """Basic UDP socket using threads for receive and callback dispatch."""
#     BUFSIZE = 65535

#     def __init__(
#         self,
#         ip_addr: str,
#         port: int,
#         callback: callable[[bytes, tuple], None],
#         *,
#         logger: logging.Logger | None = None,
#     ) -> None:
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         self.sock.bind((ip_addr, port))
#         self.sock.connect((ip_addr, port))
#         self.sock.setblocking(False)

#         self.callback = callback
#         self.logger = logger or logging.getLogger("UdpSocket")

#         self.executor: Thread | None = None

#         self.spinlock = threading.Lock()
#         self.running_event = threading.Event()

#     @property
#     def running(self):
#         return self.running.is_set()

#     def run(self) -> None:
#         with self.spinlock:
#             if self.running_event.is_set():
#                 return
#             self.running_event.set()
#             self.executor = Thread(target=self.loop, daemon=True)
#             self.executor.start()

#     def send(self, data: bytes, addr: tuple[str, int] | None = None) -> None:
#         if not self.running_event.is_set():
#             raise RuntimeError("UdpSocket is not running.")
#         self.sock.send(data)

#     def stop(self) -> None:
#         self.running_event.clear()
#         self.sock.close()

#     def loop(self) -> None:
#         while self.running_event.is_set():
#             try:
#                 data, _ = self.sock.recvfrom(self.BUFSIZE)
#             except TimeoutError:
#                 continue
#             except (OSError, KeyboardInterrupt):
#                 break
#             self.callback(data)