from struct import Struct
from dataclasses import dataclass

from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

@dataclass
class Protocol:
    msg_id: int
    topic_name: str
    msg_direction: str
    msg_type: type
    msg_format: Struct
    comm_entity: Publisher|Subscription