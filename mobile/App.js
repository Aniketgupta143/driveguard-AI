
import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  StatusBar,
} from "react-native";

const WS_URL = "wss://choosy-barcode-ferment.ngrok-free.dev/ws";

const initialData = {
  camera_status: "Connecting",
  face_detected: false,
  eye_status: "Unknown",
  ear: 0,
  drowsy: false,
  yawning: false,
  mar: 0,
  head_status: "Unknown",
  pitch_change: 0,
  yaw_change: 0,
  perclos: 0,
  perclos_observed_seconds: 0,
  last_updated: null,
};

function StatusCard({ title, value, subtitle, color = "#22c55e" }) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>{title}</Text>
      <Text style={[styles.cardValue, { color }]}>{value}</Text>
      {subtitle ? <Text style={styles.cardSubtitle}>{subtitle}</Text> : null}
    </View>
  );
}

function AlertRow({ title, active, activeText, inactiveText }) {
  return (
    <View style={styles.alertRow}>
      <View style={styles.alertInfo}>
        <View
          style={[
            styles.statusDot,
            { backgroundColor: active ? "#ef4444" : "#22c55e" },
          ]}
        />
        <Text style={styles.alertTitle}>{title}</Text>
      </View>

      <Text
        style={[
          styles.alertValue,
          { color: active ? "#ef4444" : "#22c55e" },
        ]}
      >
        {active ? activeText : inactiveText}
      </Text>
    </View>
  );
}

export default function App() {
  const [data, setData] = useState(initialData);
  const [connection, setConnection] = useState("Connecting...");
  const socketRef = useRef(null);

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let isMounted = true;

    const connect = () => {
      if (!isMounted) return;

      setConnection("Connecting...");

      socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
        if (isMounted) setConnection("Connected");
      };

      socket.onmessage = (event) => {
        try {
          const receivedData = JSON.parse(event.data);

          if (isMounted) {
            setData(receivedData);
          }
        } catch (error) {
          console.log("Invalid monitoring data:", error);
        }
      };

      socket.onerror = () => {
        if (isMounted) setConnection("Connection error");
      };

      socket.onclose = () => {
        if (!isMounted) return;

        setConnection("Disconnected");
        reconnectTimer = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      isMounted = false;
      clearTimeout(reconnectTimer);

      if (socket) {
        socket.onclose = null;
        socket.close();
      }

      socketRef.current = null;
    };
  }, []);

  const connected = connection === "Connected";
  const cameraRunning = data.camera_status === "running";

  const headAlert =
    String(data.head_status).toLowerCase().includes("alert");

  const perclosValue = Number(data.perclos) || 0;

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#080d18" />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.logo}>DRIVEGUARD AI</Text>
          <Text style={styles.tagline}>
            Intelligent Driver Monitoring System
          </Text>

          <View style={styles.connectionBox}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: connected ? "#22c55e" : "#ef4444" },
              ]}
            />
            <Text style={styles.connectionText}>
              {connection}
            </Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>SYSTEM STATUS</Text>

          <View style={styles.mainStatus}>
            <Text style={styles.mainStatusLabel}>CAMERA STATUS</Text>
            <Text
              style={[
                styles.mainStatusValue,
                { color: cameraRunning ? "#22c55e" : "#f59e0b" },
              ]}
            >
              {String(data.camera_status).toUpperCase()}
            </Text>
            <Text style={styles.smallText}>
              {data.face_detected
                ? "Driver face detected"
                : "No driver face detected"}
            </Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>DRIVER MONITORING</Text>

          <View style={styles.grid}>
            <StatusCard
              title="FACE DETECTION"
              value={data.face_detected ? "DETECTED" : "NOT FOUND"}
              color={data.face_detected ? "#22c55e" : "#f59e0b"}
            />

            <StatusCard
              title="EYE STATUS"
              value={data.eye_status || "Unknown"}
              color={
                String(data.eye_status).toLowerCase().includes("closed")
                  ? "#ef4444"
                  : "#22c55e"
              }
            />

            <StatusCard
              title="EAR VALUE"
              value={Number(data.ear || 0).toFixed(3)}
              subtitle="Eye aspect ratio"
            />

            <StatusCard
              title="YAWNING"
              value={data.yawning ? "DETECTED" : "NORMAL"}
              color={data.yawning ? "#ef4444" : "#22c55e"}
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>SAFETY ALERTS</Text>

          <View style={styles.alertPanel}>
            <AlertRow
              title="Drowsiness"
              active={data.drowsy}
              activeText="WARNING"
              inactiveText="NORMAL"
            />

            <View style={styles.separator} />

            <AlertRow
              title="Yawning"
              active={data.yawning}
              activeText="DETECTED"
              inactiveText="NORMAL"
            />

            <View style={styles.separator} />

            <AlertRow
              title="Head Position"
              active={headAlert}
              activeText="WARNING"
              inactiveText="NORMAL"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>HEAD POSITION ANALYSIS</Text>

          <View style={styles.grid}>
            <StatusCard
              title="HEAD STATUS"
              value={data.head_status || "Unknown"}
              color={headAlert ? "#ef4444" : "#22c55e"}
            />

            <StatusCard
              title="PITCH CHANGE"
              value={`${Number(data.pitch_change || 0).toFixed(2)}°`}
            />

            <StatusCard
              title="YAW CHANGE"
              value={`${Number(data.yaw_change || 0).toFixed(2)}°`}
            />

            <StatusCard
              title="MAR VALUE"
              value={Number(data.mar || 0).toFixed(3)}
              subtitle="Mouth aspect ratio"
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>PERCLOS ANALYSIS</Text>

          <View style={styles.perclosCard}>
            <Text style={styles.perclosValue}>
              {perclosValue.toFixed(2)}%
            </Text>

            <Text style={styles.cardSubtitle}>
              Estimated percentage of time eyes were closed
            </Text>

            <View style={styles.progressBackground}>
              <View
                style={[
                  styles.progressFill,
                  {
                    width: `${Math.min(100, Math.max(0, perclosValue))}%`,
                  },
                ]}
              />
            </View>

            <Text style={styles.smallText}>
              Observation window:{" "}
              {Number(data.perclos_observed_seconds || 0).toFixed(0)} seconds
            </Text>
          </View>
        </View>

        <Text style={styles.footer}>
          DRIVEGUARD AI • DRIVER SAFETY MONITORING PROTOTYPE
        </Text>
        <Text style={styles.disclaimer}>
          Experimental prototype. Not validated for real-world driving safety.
        </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#080d18",
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    marginTop: 35,
    marginBottom: 28,
  },
  logo: {
    fontSize: 27,
    fontWeight: "900",
    color: "#38bdf8",
    letterSpacing: 2,
  },
  tagline: {
    color: "#94a3b8",
    fontSize: 13,
    marginTop: 6,
  },
  connectionBox: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    backgroundColor: "#111c2e",
    paddingVertical: 9,
    paddingHorizontal: 13,
    borderRadius: 20,
    marginTop: 18,
  },
  connectionText: {
    color: "#e2e8f0",
    marginLeft: 8,
    fontSize: 12,
    fontWeight: "600",
  },
  section: {
    marginBottom: 25,
  },
  sectionTitle: {
    color: "#38bdf8",
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 1.5,
    marginBottom: 13,
  },
  mainStatus: {
    backgroundColor: "#111c2e",
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: "#23334a",
  },
  mainStatusLabel: {
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: "700",
  },
  mainStatusValue: {
    fontSize: 23,
    fontWeight: "900",
    marginTop: 8,
  },
  smallText: {
    color: "#94a3b8",
    fontSize: 11,
    marginTop: 9,
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    gap: 12,
  },
  card: {
    width: "48%",
    minHeight: 112,
    backgroundColor: "#111c2e",
    borderRadius: 14,
    padding: 15,
    borderWidth: 1,
    borderColor: "#23334a",
  },
  cardTitle: {
    color: "#94a3b8",
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  cardValue: {
    fontSize: 17,
    fontWeight: "900",
    marginTop: 13,
  },
  cardSubtitle: {
    color: "#94a3b8",
    fontSize: 11,
    marginTop: 7,
  },
  alertPanel: {
    backgroundColor: "#111c2e",
    borderRadius: 14,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: "#23334a",
  },
  alertRow: {
    minHeight: 57,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  alertInfo: {
    flexDirection: "row",
    alignItems: "center",
  },
  statusDot: {
    width: 9,
    height: 9,
    borderRadius: 10,
  },
  alertTitle: {
    color: "#e2e8f0",
    fontSize: 13,
    marginLeft: 10,
    fontWeight: "600",
  },
  alertValue: {
    fontSize: 11,
    fontWeight: "800",
  },
  separator: {
    height: 1,
    backgroundColor: "#23334a",
  },
  perclosCard: {
    backgroundColor: "#111c2e",
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: "#23334a",
  },
  perclosValue: {
    color: "#38bdf8",
    fontSize: 34,
    fontWeight: "900",
  },
  progressBackground: {
    height: 9,
    backgroundColor: "#263449",
    borderRadius: 10,
    marginTop: 18,
    overflow: "hidden",
  },
  progressFill: {
    height: 9,
    backgroundColor: "#38bdf8",
    borderRadius: 10,
  },
  footer: {
    color: "#64748b",
    fontSize: 10,
    textAlign: "center",
    fontWeight: "800",
    marginTop: 10,
  },
  disclaimer: {
    color: "#64748b",
    fontSize: 10,
    textAlign: "center",
    marginTop: 8,
  },
});
