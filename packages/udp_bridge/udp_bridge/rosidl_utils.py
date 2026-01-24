import os
from collections import OrderedDict

from ament_index_python import get_package_share_directory
from rosidl_parser.definition import (
    BasicType,
    NamespacedType,
    AbstractNestedType,
    Array,
)
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py import (
    message_to_ordereddict,
    set_message_fields
)

message_type_cache = {}
format_cache = {}

BASIC_TYPE_FORMAT = {
    "boolean": "?",
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "short": "h",
    "unsigned short": "H",
    "long": "l",
    "unsigned long": "L",
    "long long": "q",
    "unsigned long long": "Q",
    "float": "f",
    "double": "d",
    "octet": "B",
    "char": "b",
    "wchar": "I",
}

def parse_url(url):
    if url.startswith("package://"):
        package_and_path = url[len("package://"):]
        package, _, relative_path = package_and_path.partition("/")
        if package:
            base_dir = get_package_share_directory(package)
            return os.path.join(base_dir, relative_path) if relative_path else base_dir
    if url.startswith("file://"):
        return url[len("file://"):]
    return url


def fill_message_from_values(msg, values):
    it = iter(values)

    def fill_message_from_values_impl(node):
        out = OrderedDict()
        for k, v in node.items():
            if isinstance(v, OrderedDict):
                out[k] = fill_message_from_values_impl(v)
            else:
                out[k] = next(it)
        return out

    template = fill_message_from_values_impl(message_to_ordereddict(msg))

    try:
        next(it)
        raise ValueError("Too many values: unpacked data longer than template leaf count")
    except StopIteration:
        pass

    set_message_fields(msg, template)

    return msg

def flatten_message_to_values(msg):
    out = []
    def flatten_message_to_values_impl(node):
        for _, v in node.items():
            if isinstance(v, OrderedDict):
                flatten_message_to_values_impl(v)
            else:
                out.append(v)
    flatten_message_to_values_impl(message_to_ordereddict(msg))
    return out

def message_type_to_struct_format(msg_type, prefix=None):
    if prefix is None:
        prefix = ""

    parts = []
    for slot_type in msg_type.SLOT_TYPES:
        parts.append(format_for_slot_type(slot_type))
    return prefix + "".join(parts)

def format_for_slot_type(slot_type):
    if isinstance(slot_type, BasicType):
        try:
            return BASIC_TYPE_FORMAT[slot_type.typename]
        except KeyError as exc:
            raise ValueError(f"Unsupported basic type {slot_type.typename}") from exc
    if isinstance(slot_type, NamespacedType):
        type_name = "/".join((*slot_type.namespaces, slot_type.name))
        cached = format_cache.get(type_name)
        if cached is not None:
            return cached
        sub_msg_type = message_type_cache.get(type_name)
        if sub_msg_type is None:
            sub_msg_type = get_message(type_name)
            message_type_cache[type_name] = sub_msg_type
        sub_format = message_type_to_struct_format(sub_msg_type)
        format_cache[type_name] = sub_format
        return sub_format
    if isinstance(slot_type, AbstractNestedType):
        if not isinstance(slot_type, Array):
            raise ValueError("Sequences are not supported for struct format.")
        return format_for_slot_type(slot_type.value_type) * slot_type.size
    raise ValueError(f"Unsupported slot type {type(slot_type)}")

