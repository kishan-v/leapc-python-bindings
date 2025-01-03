import time
from leap import datatypes as ldt
import leap
import math
from typing import Tuple

import leap.events


def quaternion_to_euler(
    x: float, y: float, z: float, w: float
) -> Tuple[float, float, float]:
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    Roll is rotation around the x-axis, pitch is rotation around the y-axis,
    and yaw is rotation around the z-axis.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_relative_rotation(
    bone1: ldt.Bone, bone2: ldt.Bone
) -> Tuple[float, float, float]:
    """
    Calculate the relative rotation between two bones and return as Euler angles.
    """
    q1 = bone1.rotation
    q2 = bone2.rotation

    # Inverse of q1
    q1_inv = ldt.Quaternion(-q1.x, -q1.y, -q1.z, q1.w)

    # Quaternion multiplication (q2 * q1_inv)
    qr_w = q2.w * q1_inv.w - q2.x * q1_inv.x - q2.y * q1_inv.y - q2.z * q1_inv.z
    qr_x = q2.w * q1_inv.x + q2.x * q1_inv.w + q2.y * q1_inv.z - q2.z * q1_inv.y
    qr_y = q2.w * q1_inv.y - q2.x * q1_inv.z + q2.y * q1_inv.w + q2.z * q1_inv.x
    qr_z = q2.w * q1_inv.z + q2.x * q1_inv.y - q2.y * q1_inv.x + q2.z * q1_inv.w

    return quaternion_to_euler(qr_x, qr_y, qr_z, qr_w)


class JointAngleListener(leap.Listener):
    def on_tracking_event(self, event: leap.events.TrackingEvent):
        if event.tracking_frame_id % 50 == 0:
            for hand in event.hands:
                hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
                print(f"{hand_type} hand:")

                # Iterate through each finger
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                fingers = [hand.thumb, hand.index, hand.middle, hand.ring, hand.pinky]

                for name, digit in zip(finger_names, fingers):
                    print(f"  {name} finger:")

                    # Get bone vectors
                    bones = digit.bones
                    for i in range(len(bones) - 1):
                        # Get vectors for consecutive bones
                        bone1 = bones[i]
                        bone2 = bones[i + 1]

                        # Calculate vectors from bone endpoints
                        vec1 = [
                            bone1.next_joint.x - bone1.prev_joint.x,
                            bone1.next_joint.y - bone1.prev_joint.y,
                            bone1.next_joint.z - bone1.prev_joint.z,
                        ]

                        vec2 = [
                            bone2.next_joint.x - bone2.prev_joint.x,
                            bone2.next_joint.y - bone2.prev_joint.y,
                            bone2.next_joint.z - bone2.prev_joint.z,
                        ]

                        # Calculate angle using dot product
                        dot_product = sum(a * b for a, b in zip(vec1, vec2))
                        magnitude1 = sum(x * x for x in vec1) ** 0.5
                        magnitude2 = sum(x * x for x in vec2) ** 0.5

                        if magnitude1 * magnitude2 == 0:
                            angle = 0
                        else:
                            angle = math.acos(dot_product / (magnitude1 * magnitude2))
                            angle = math.degrees(angle)

                        bone_names = [
                            "Metacarpal-Proximal",
                            "Proximal-Intermediate",
                            "Intermediate-Distal",
                        ]
                        print(f"    {bone_names[i]} angle: {angle:.1f}°")
                print()


def main():
    listener = JointAngleListener()

    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
