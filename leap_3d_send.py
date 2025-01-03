import leap
import leap.datatypes
import leap.events
import socket
import time
import cv2
from examples.visualiser import Canvas, TrackingListener


_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}


class HandDataTransmitter:
    def __init__(self, host="127.0.0.1", port=5052):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = (host, port)

    def format_hand_data(self, hand: leap.datatypes.Hand) -> str:
        points = []

        # 0: WRIST
        wrist = hand.arm.next_joint
        points.extend([wrist.x, wrist.y, wrist.z])

        # THUMB (1-4)
        # 1: CMC - Get from thumb metacarpal base
        thumb_cmc = hand.thumb.metacarpal.prev_joint
        points.extend([thumb_cmc.x, thumb_cmc.y, thumb_cmc.z])

        # 2: MCP - Metacarpal tip/proximal base
        thumb_mcp = hand.thumb.proximal.prev_joint
        points.extend([thumb_mcp.x, thumb_mcp.y, thumb_mcp.z])

        # 3: IP - Proximal tip/distal base
        thumb_ip = hand.thumb.intermediate.prev_joint
        points.extend([thumb_ip.x, thumb_ip.y, thumb_ip.z])

        # 4: TIP - Distal tip
        thumb_tip = hand.thumb.distal.next_joint
        points.extend([thumb_tip.x, thumb_tip.y, thumb_tip.z])

        # Process fingers (index, middle, ring, pinky)
        fingers = [hand.index, hand.middle, hand.ring, hand.pinky]
        for finger in fingers:
            # MCP joint (5,9,13,17) - Metacarpal tip
            mcp = finger.metacarpal.next_joint
            points.extend([mcp.x, mcp.y, mcp.z])

            # PIP joint (6,10,14,18) - Proximal tip
            pip = finger.proximal.next_joint
            points.extend([pip.x, pip.y, pip.z])

            # DIP joint (7,11,15,19) - Intermediate tip
            dip = finger.intermediate.next_joint
            points.extend([dip.x, dip.y, dip.z])

            # Fingertip (8,12,16,20) - Distal tip
            tip = finger.distal.next_joint
            points.extend([tip.x, tip.y, tip.z])

        assert len(points) == 21 * 3
        # Scale all points from mm to meters
        points = [p / 1000 for p in points]

        return "[" + ",".join(map(str, points)) + "]"

    def send_data(self, data: str):
        self.sock.sendto(data.encode(), self.address)


class HandTrackingListener(TrackingListener):
    def __init__(self, transmitter: HandDataTransmitter, canvas: Canvas):
        self.transmitter = transmitter
        self.canvas = canvas

    def on_tracking_event(self, event: leap.events.TrackingEvent):
        self.canvas.render_hands(event)  # TODO: threading
        for hand in event.hands:
            if hand.type == leap.datatypes.HandType.Left:
                data = self.transmitter.format_hand_data(hand)
                self.transmitter.send_data(data)  # TODO: threading
                break  # Only send left hand data

    def on_tracking_mode_event(self, event: leap.events.TrackingModeEvent):
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        print(
            f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}"
        )


def main():
    transmitter = HandDataTransmitter()
    canvas = Canvas()
    listener = HandTrackingListener(transmitter, canvas)

    connection = leap.Connection()
    connection.add_listener(listener)

    print("Starting UDP transmitter")
    print("Press 'x' in visualizer window to quit")

    running = True

    try:
        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            canvas.set_tracking_mode(leap.TrackingMode.Desktop)

            while running:
                cv2.imshow(canvas.name, canvas.output_image)

                key = cv2.waitKey(1)

                if key == ord("x"):
                    break
                elif key == ord("h"):
                    connection.set_tracking_mode(leap.TrackingMode.HMD)
                elif key == ord("s"):
                    connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
                elif key == ord("d"):
                    connection.set_tracking_mode(leap.TrackingMode.Desktop)
                elif key == ord("f"):
                    canvas.toggle_hands_format()

                # time.sleep(0.01)  # Small delay to prevent overwhelming the network

    except KeyboardInterrupt:
        print("\nShutting down...")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
