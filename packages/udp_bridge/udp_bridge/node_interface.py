import yaml
import struct
from abc import abstractmethod, ABC
from functools import partial, lru_cache

from rclpy.logging import RcutilsLogger
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosidl_runtime_py.utilities import get_message

from .protocol import Protocol
from .rosidl_utils import (
    fill_message_from_values,
    flatten_message_to_values,
    message_type_to_struct_format,
    parse_url
)
from .udp_socket import UdpSocket

class UdpBridgeNodeInterface(Node, ABC):
    node_name = "udp_bridge"
    def __init__(self):
        super().__init__(
            self.node_name, 
            automatically_declare_parameters_from_overrides=True
        )

        self.sock = UdpSocket(
            local_ip=self.get_parameter("local_ip").value,
            local_port=self.get_parameter("local_port").value,
            remote_ip=self.get_parameter("remote_ip").value,
            remote_port=self.get_parameter("remote_port").value,
            timeout=self.get_parameter("timeout").value,
            logger=self.logger
        )

        self._init_protocol()

        self.sock.on_message(self.uplink_callback)
        self.sock.start()

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()
    
    @abstractmethod
    def downlink_callback(self, msg, topic_name=None):
        raise NotImplementedError()
    
    @abstractmethod
    def uplink_callback(self, data, msg_id=None):
        raise NotImplementedError()

    def destroy_node(self):
        self.sock.stop()
        return super().destroy_node()

    @lru_cache
    def get_protocol(self, topic_name_or_msg_id: str|int) -> Protocol|None:
        if isinstance(topic_name_or_msg_id, int):
            for prot in self.PROTOCOL:
                if prot.msg_id == topic_name_or_msg_id:
                    return prot
            return None
        elif isinstance(topic_name_or_msg_id, str):
            for prot in self.PROTOCOL:
                if prot.topic_name == topic_name_or_msg_id:
                    return prot
            return None
        else:
            return None
    

    def _init_protocol(self):
        # magic, ver, seq, stamp, message_id
        self.sequence_number = 0

        with open(parse_url(self.get_parameter("url").value), "r", encoding="utf-8") as f:
            protocol = yaml.safe_load(f)
            self.magic_number = protocol["magic_number"]
            self.version = protocol["version"]
            self.header_format =  struct.Struct(protocol["header_format"])
            self.header_size = struct.calcsize(protocol["header_format"])

        self.PROTOCOL : list[Protocol] = []
        for ch in protocol["channels"]:
            topic_name = ch["topic"]
            msg_id = ch["id"]
            msg_direction = ch["direction"]
            msg_type = get_message(ch["type"])
            msg_format = struct.Struct(
                message_type_to_struct_format(msg_type, prefix="!")) # big endian
            
            comm_entity = None

            if ch["direction"] == "uplink":
                comm_entity = self.create_publisher(
                    msg_type,
                    ch["topic"],
                    10
                )
            elif ch["direction"] == "downlink":
                comm_entity = self.create_subscription(
                    msg_type,
                    ch["topic"],
                    10,
                    partial(self.downlink_callback, topic_name=topic_name)
                )
            else:
                raise ValueError("key `direction` must be `uplink` or `downlink`")

            self.PROTOCOL.append(Protocol(
                msg_id=msg_id,
                topic_name=topic_name,
                msg_type=msg_type,
                msg_direction=msg_direction,
                msg_format=msg_format,
                comm_entity=comm_entity
            ))


class UdpBridge(UdpBridgeNodeInterface):
    def downlink_callback(self, msg, topic_name=None):
        protocol = self.get_protocol(topic_name)
        if protocol is None:
            self.logger.error(f"Unknown topic name: {topic_name}")
            return

        stamp = self.get_clock().now().nanoseconds

        header = self.header_format.pack(
            self.magic_number,
            self.version,
            self.sequence_number,
            stamp,
            protocol.msg_id
        )

        payload = protocol.msg_format.pack(*flatten_message_to_values(msg))
        data = header + payload

        self.sock.send(data)
        self.sequence_number += 1

    def uplink_callback(self, data, msg_id=None):
        # NOTE: !HBIH -> big endian, uint16, uint8, uint32, uint16

        if len(data) < self.header_size:
            self.logger.error(
                "Got unexpected EOF while parsing datapack.",
                throttle_duration_sec=1.0)
            return

        magic_number, version, sequence_number, message_id = self.header_format.unpack_from(
            buffer=data,
            offset=0
        )

        if msg_id is not None:
            message_id = msg_id # override

        protocol = self.get_protocol(message_id)
        if protocol is None:
            self.logger.error(f"Unknown message id: {message_id}")
            return

        if version != self.version:
            self.logger.error("Version mismatch.")
            return

        if magic_number != self.magic_number:
            self.logger.error("Datapack transfer error: magic number mismatch.")
            return

        msg = fill_message_from_values(
            protocol.msg_type(),
            protocol.msg_format.unpack_from(
                buffer=data,
                offset=self.header_size))
        
        publisher = protocol.comm_entity
        publisher.publish(msg)

        # TODO: check sequence number, calculate loss pack rate

        # TODO: check timestamp, calculate transfer delay

