import argparse
import csv
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import leap
import leap.datatypes
import leap.events

from examples.visualiser import Canvas, TrackingListener

_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}


class HandDataTransmitter:
    def __init__(
        self,
        host="127.0.0.1",
        port=5052,
        sampling_rate=200,
        write_to_csv=True,
        send_over_udp=True,
        selected_hand=leap.HandType.Left,
    ):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = (host, port)
        self.write_to_csv = write_to_csv
        self.send_over_udp = send_over_udp
        self.selected_hand = selected_hand
        print(f"Ignoring hands other than {selected_hand}")

        # Sampling control
        self.sampling_interval = 1.0 / sampling_rate
        self.last_sample_time = 0

        if write_to_csv:
            # Create data directory and CSV file
            self.data_dir = Path("data")
            self.data_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = self.data_dir / f"hand_tracking_{timestamp}.csv"

            # Initialize CSV with headers
            headers = []
            landmarks = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP"]
            for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]:
                landmarks.extend(
                    [f"{finger}_MCP", f"{finger}_PIP", f"{finger}_DIP", f"{finger}_TIP"]
                )

            for landmark in landmarks:
                headers.extend([f"{landmark}_X", f"{landmark}_Y", f"{landmark}_Z"])

            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        self.frame_times = []
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0  # Update FPS display every second

        # Interpolation
        self.last_known_points = None
        self.last_frame_time = None

    def format_hand_data(self, hand: leap.datatypes.Hand) -> list[float]:
        """
        Normalised points relative to wrist position
        """
        points = []

        # 0: WRIST
        wrist = hand.arm.next_joint
        # points.extend([wrist.x, wrist.y, wrist.z])
        points.extend([0, 0, 0])

        # THUMB (1-4)
        # 1: CMC - Get from thumb metacarpal base
        thumb_cmc = hand.thumb.metacarpal.prev_joint
        points.extend([thumb_cmc.x - wrist.x, thumb_cmc.y - wrist.y, thumb_cmc.z - wrist.z])

        # 2: MCP - Metacarpal tip/proximal base
        thumb_mcp = hand.thumb.proximal.prev_joint
        points.extend([thumb_mcp.x - wrist.x, thumb_mcp.y - wrist.y, thumb_mcp.z - wrist.z])

        # 3: IP - Proximal tip/distal base
        thumb_ip = hand.thumb.intermediate.prev_joint
        points.extend([thumb_ip.x - wrist.x, thumb_ip.y - wrist.y, thumb_ip.z - wrist.z])

        # 4: TIP - Distal tip
        thumb_tip = hand.thumb.distal.next_joint
        points.extend([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y, thumb_tip.z - wrist.z])

        # Process fingers (index, middle, ring, pinky)
        fingers = [hand.index, hand.middle, hand.ring, hand.pinky]
        for finger in fingers:
            # MCP joint (5,9,13,17) - Metacarpal tip
            mcp = finger.metacarpal.next_joint
            points.extend([mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z])

            # PIP joint (6,10,14,18) - Proximal tip
            pip = finger.proximal.next_joint
            points.extend([pip.x - wrist.x, pip.y - wrist.y, pip.z - wrist.z])

            # DIP joint (7,11,15,19) - Intermediate tip
            dip = finger.intermediate.next_joint
            points.extend([dip.x - wrist.x, dip.y - wrist.y, dip.z - wrist.z])

            # Fingertip (8,12,16,20) - Distal tip
            tip = finger.distal.next_joint
            points.extend([tip.x - wrist.x, tip.y - wrist.y, tip.z - wrist.z])

        # Scale all points from mm to meters
        scaled_points = [p / 1000 for p in points]

        return scaled_points

    def send_data(self, data: str):
        self.sock.sendto(data.encode(), self.address)

    def interpolate_points(self, current_points: list[float], num_samples: int) -> list[list]:
        if self.last_known_points is None:
            # For first frame, duplicate current points
            return [current_points] * num_samples

        interpolated = []
        for i in range(num_samples):
            fraction = (i + 1) / (num_samples + 1)
            sample = []
            # Linear interpolation between last and current points
            for last, current in zip(self.last_known_points, current_points):
                interpolated_value = last + (current - last) * fraction
                sample.append(interpolated_value)
            interpolated.append(sample)
        return interpolated

    def handle_data(self, hand: leap.datatypes.Hand):
        if hand.type != self.selected_hand:
            return

        current_time = time.time()
        self.frame_times.append(current_time)

        # Handle FPS display
        if current_time - self.last_fps_update >= self.fps_update_interval:
            fps = len([t for t in self.frame_times if t > current_time - 1])
            sys.stdout.write(f"\rFrame Rate: {fps:3d} FPS    ")
            sys.stdout.flush()
            self.last_fps_update = current_time

        # Get current hand position
        current_points = self.format_hand_data(hand)

        # Calculate missed samples
        if self.last_frame_time is not None:
            time_gap = current_time - self.last_frame_time
            missed_samples = int(time_gap / self.sampling_interval) - 1

            if missed_samples > 0:
                interpolated = self.interpolate_points(current_points, missed_samples)

                # Write interpolated samples
                for sample in interpolated:
                    if self.write_to_csv:
                        with open(self.csv_path, "a", newline="") as f:
                            csv.writer(f).writerow(sample)

                    if self.send_over_udp:
                        payload = "[" + ",".join(map(str, sample)) + "]"
                        self.send_data(payload)

        # Write current sample
        if time.time() - self.last_sample_time > self.sampling_interval:
            if self.write_to_csv:
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(current_points)

            if self.send_over_udp:
                payload = "[" + ",".join(map(str, current_points)) + "]"
                self.send_data(payload)

            self.last_sample_time = time.time()

        # Update last known state
        self.last_known_points = current_points
        self.last_frame_time = current_time


class HandTrackingListener(TrackingListener):
    def __init__(self, transmitter: HandDataTransmitter, canvas: Canvas):
        self.transmitter = transmitter
        self.canvas = canvas

    def on_tracking_event(self, event: leap.events.TrackingEvent):
        self.canvas.render_hands(event)  # TODO: threading
        for hand in event.hands:
            self.transmitter.handle_data(hand)
            break  # Only send left hand data

    def on_tracking_mode_event(self, event: leap.events.TrackingModeEvent):
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        print(
            f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leap Motion hand tracking data handler"
    )
    parser.add_argument("mode", choices=["record", "playback"], help="Operating mode")

    # Common options
    parser.add_argument(
        "--write-csv", action="store_true", default=True, help="Write data to CSV"
    )
    parser.add_argument(
        "--send-udp", action="store_true", default=True, help="Send data over UDP"
    )
    parser.add_argument(
        "--udp-port", type=int, default=5052, help="UDP port (default: 5052)"
    )
    parser.add_argument(
        "--udp-host", default="127.0.0.1", help="UDP host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=200,
        help="Sampling rate in Hz (default: 200)",
    )
    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default="left",
        help="Hand to track (default: left)",
    )
    parser.add_argument(
        "--csv-data",
        type=str,
        help="Path to CSV file for playback mode",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"Operating mode: {args.mode}\nSelected hand: {args.hand}\nSampling rate: {args.sampling_rate} Hz"
    )

    if args.mode == "record":
        transmitter = HandDataTransmitter(
            host=args.udp_host,
            port=args.udp_port,
            write_to_csv=args.write_csv,
            send_over_udp=args.send_udp,
            sampling_rate=args.sampling_rate,
            selected_hand=leap.HandType.Left
            if args.hand == "left"
            else leap.HandType.Right,
        )
        canvas = Canvas()

        listener = HandTrackingListener(transmitter, canvas)
        connection = leap.Connection()
        connection.add_listener(listener)

        print(f"Starting UDP transmitter in {args.mode} mode")
        print("Press 'q' in visualizer window to quit\n")

        running = True

        try:
            with connection.open():
                connection.set_tracking_mode(leap.TrackingMode.Desktop)
                canvas.set_tracking_mode(leap.TrackingMode.Desktop)

                while running:
                    cv2.imshow(canvas.name, canvas.output_image)

                    key = cv2.waitKey(1)

                    if key == ord("q"):
                        break
                    elif key == ord("h"):
                        connection.set_tracking_mode(leap.TrackingMode.HMD)
                    elif key == ord("s"):
                        connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
                    elif key == ord("d"):
                        connection.set_tracking_mode(leap.TrackingMode.Desktop)
                    elif key == ord("f"):
                        canvas.toggle_hands_format()

        except KeyboardInterrupt:
            print("\nShutting down...")

        cv2.destroyAllWindows()

    elif args.mode == "playback":
        transmitter = HandDataTransmitter(
            host=args.udp_host,
            port=args.udp_port,
            write_to_csv=False,
            send_over_udp=True,
            sampling_rate=args.sampling_rate,
        )

        # Add csv_data argument check
        if args.csv_data is None:
            raise Exception("Playback mode requires --csv-data argument")

        print(f"Starting UDP transmitter in {args.mode} mode")
        print(f"Reading from {args.csv_data}")

        try:
            with open(args.csv_data, "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row

                for row in reader:
                    if not row:
                        continue
                    data = [float(x) for x in row]
                    if args.send_udp:
                        payload = "[" + ",".join(map(str, data)) + "]"
                        transmitter.send_data(payload)

                    time.sleep(transmitter.sampling_interval)
            print("Playback complete")
        except KeyboardInterrupt:
            print("\nShutting down...")

    else:
        print("Invalid mode. Use 'record' or 'playback'")


if __name__ == "__main__":
    main()
