import leap
import leap.datatypes
import leap.events
import numpy as np
import cv2
import math


class AngleVisualiser:
    def __init__(self):
        self.name = "Hand Angle Visualiser"
        self.screen_size = [500, 700]
        self.skeleton_color = (255, 255, 255)
        self.text_color = (0, 255, 0)
        self.output_image = np.zeros(
            (self.screen_size[0], self.screen_size[1], 3), np.uint8
        )

    def get_joint_position(self, bone: leap.datatypes.Bone):
        if bone:
            return int(bone.x + (self.screen_size[1] / 2)), int(
                bone.z + (self.screen_size[0] / 2)
            )
        return None

    def calculate_angle(self, bone1: leap.datatypes.Bone, bone2: leap.datatypes.Bone):
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

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(x * x for x in vec1) ** 0.5
        magnitude2 = sum(x * x for x in vec2) ** 0.5

        if magnitude1 * magnitude2 == 0:
            return 0
        angle = math.acos(dot_product / (magnitude1 * magnitude2))
        return math.degrees(angle)

    def render_hand(self, hand: leap.datatypes.Hand):
        # Draw skeleton
        for digit in hand.digits:
            for i in range(len(digit.bones)):
                bone = digit.bones[i]
                start_pos = self.get_joint_position(bone.prev_joint)
                end_pos = self.get_joint_position(bone.next_joint)

                if start_pos and end_pos:
                    cv2.line(
                        self.output_image, start_pos, end_pos, self.skeleton_color, 2
                    )
                    cv2.circle(self.output_image, start_pos, 3, self.skeleton_color, -1)
                    cv2.circle(self.output_image, end_pos, 3, self.skeleton_color, -1)

                # Draw angles between consecutive bones
                if i < len(digit.bones) - 1:
                    next_bone = digit.bones[i + 1]
                    angle = self.calculate_angle(bone, next_bone)
                    mid_x = (start_pos[0] + end_pos[0]) // 2
                    mid_y = (start_pos[1] + end_pos[1]) // 2
                    cv2.putText(
                        self.output_image,
                        f"{angle:.1f}°",
                        (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        self.text_color,
                        1,
                    )


class AngleTrackingListener(leap.Listener):
    def __init__(self, visualiser: AngleVisualiser):
        self.visualiser = visualiser

    def on_tracking_event(self, event: leap.events.TrackingEvent):
        self.visualiser.output_image[:] = 0  # Clear frame
        for hand in event.hands:
            self.visualiser.render_hand(hand)


def main():
    visualiser = AngleVisualiser()
    listener = AngleTrackingListener(visualiser)

    connection = leap.Connection()
    connection.add_listener(listener)

    print("Press 'q' to quit")

    with connection.open():
        while True:
            cv2.imshow(visualiser.name, visualiser.output_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
