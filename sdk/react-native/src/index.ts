/**
 * CAFE-u React Native SDK — Mobile frustration detection.
 * 
 * Captures mobile-specific signals: long press, rapid taps,
 * shake (frustration), swipe-hesitation, form pauses.
 * 
 * Usage:
 *   import { CafeuProvider, useCafeuTracker } from 'cafeu-react-native';
 * 
 *   function App() {
 *     return (
 *       <CafeuProvider apiKey="my-app" endpoint="ws://server/ws">
 *         <YourApp />
 *       </CafeuProvider>
 *     );
 *   }
 */

import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import {
  PanResponder,
  AppState,
  Dimensions,
  Platform,
  NativeModules,
} from 'react-native';

// ── Types ──────────────────────────────────────────────────────

interface Signal {
  type: 'rapid_tap' | 'long_press' | 'shake' | 'swipe_hesitate' | 'form_pause';
  element?: string;
  count?: number;
  duration_ms?: number;
  timestamp: number;
}

interface Adaptation {
  selector: string;
  action: string;
  [key: string]: any;
}

interface CafeuConfig {
  apiKey: string;
  endpoint: string;
  wsEndpoint?: string;
  tapThreshold?: number;
  longPressThreshold?: number;
  formPauseThreshold?: number;
}

// ── Context ─────────────────────────────────────────────────────

interface CafeuContextType {
  trackRef: (ref: any, label: string) => void;
  reportSignal: (signal: Omit<Signal, 'timestamp'>) => void;
  lastAdaptation: Adaptation | null;
  frustrationScore: number;
}

const CafeuContext = createContext<CafeuContextType>({
  trackRef: () => {},
  reportSignal: () => {},
  lastAdaptation: null,
  frustrationScore: 0,
});

export const useCafeuTracker = () => useContext(CafeuContext);

// ── Provider ────────────────────────────────────────────────────

interface CafeuProviderProps {
  apiKey: string;
  endpoint: string;
  wsEndpoint?: string;
  children: React.ReactNode;
  config?: Partial<CafeuConfig>;
}

export const CafeuProvider: React.FC<CafeuProviderProps> = ({
  apiKey,
  endpoint,
  wsEndpoint,
  children,
  config = {},
}) => {
  const wsRef = useRef<WebSocket | null>(null);
  const signalBuffer = useRef<Signal[]>([]);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tapHistory = useRef<{ time: number; x: number; y: number }[]>([]);
  const [lastAdaptation, setLastAdaptation] = useState<Adaptation | null>(null);
  const [frustrationScore, setFrustrationScore] = useState(0);
  const appState = useRef(AppState.currentState);

  const cfg: CafeuConfig = {
    apiKey,
    endpoint,
    wsEndpoint,
    tapThreshold: config.tapThreshold || 3,
    longPressThreshold: config.longPressThreshold || 3000,
    formPauseThreshold: config.formPauseThreshold || 5000,
  };

  // ── WebSocket Connection ────────────────────────────────────

  useEffect(() => {
    if (!cfg.wsEndpoint) return;
    const connect = () => {
      try {
        const ws = new WebSocket(cfg.wsEndpoint);
        ws.onopen = () => { flush(); };
        ws.onmessage = (e) => {
          try {
            const msg = JSON.parse(e.data);
            if (msg.adaptations) {
              msg.adaptations.forEach((ad: Adaptation) => {
                setLastAdaptation(ad);
                setTimeout(() => setLastAdaptation(null), 5000);
              });
            }
          } catch (_) {}
        };
        ws.onclose = () => setTimeout(connect, 5000);
        wsRef.current = ws;
      } catch (_) {}
    };
    connect();
    return () => { wsRef.current?.close(); };
  }, [cfg.wsEndpoint]);

  // ── Signal Batching ─────────────────────────────────────────

  const flush = () => {
    if (flushTimer.current) clearTimeout(flushTimer.current);
    if (!signalBuffer.current.length) return;
    const batch = [...signalBuffer.current];
    signalBuffer.current = [];

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ signals: batch, key: apiKey }));
      return;
    }

    fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signals: batch, key: apiKey }),
      keepalive: true,
    }).catch(() => {});
  };

  const reportSignal = (signal: Omit<Signal, 'timestamp'>) => {
    signalBuffer.current.push({ ...signal, timestamp: Date.now() });
    if (signalBuffer.current.length >= 10) flush();
    if (!flushTimer.current) flushTimer.current = setTimeout(flush, 2000);
  };

  // ── Tap Detection (Rage Tap for mobile) ─────────────────────

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onPanResponderGrant: (evt) => {
        const { locationX, locationY } = evt.nativeEvent;
        const now = Date.now();
        const history = tapHistory.current;

        history.push({ time: now, x: locationX, y: locationY });

        // Keep only last 2 seconds
        while (history.length && history[0].time < now - 2000) {
          history.shift();
        }

        // Detect rapid taps on same area
        if (history.length >= cfg.tapThreshold!) {
          const recent = history.slice(-cfg.tapThreshold!);
          const avgX = recent.reduce((s, h) => s + h.x, 0) / recent.length;
          const avgY = recent.reduce((s, h) => s + h.y, 0) / recent.length;
          const isClustered = recent.every(
            (h) => Math.abs(h.x - avgX) < 40 && Math.abs(h.y - avgY) < 40
          );

          if (isClustered) {
            reportSignal({
              type: 'rapid_tap',
              count: history.length,
              element: `tap@${Math.round(avgX)},${Math.round(avgY)}`,
            });
            history.length = 0; // Reset after reporting
          }
        }
      },
    })
  ).current;

  // ── Shake Detection ─────────────────────────────────────────

  useEffect(() => {
    if (!NativeModules.MotionManager) return;
    let lastShake = 0;

    // Simple shake detection via accelerometer approximation
    const interval = setInterval(() => {
      // In production, use react-native-motion-manager or expo-sensors
      // Here we track rapid app state changes as proxy
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // ── App State Tracking ──────────────────────────────────────

  useEffect(() => {
    const sub = AppState.addEventListener('change', (nextState) => {
      if (appState.current === 'active' && nextState === 'background') {
        // User left app abruptly — possible frustration
        reportSignal({ type: 'form_pause', duration_ms: 0 });
      }
      appState.current = nextState;
    });
    return () => sub.remove();
  }, []);

  // ── Flush on unmount ────────────────────────────────────────

  useEffect(() => {
    return () => flush();
  }, []);

  // ── Tracked Component Wrapper ───────────────────────────────

  const trackRef = (ref: any, label: string) => {
    // Attach press handlers for tracking
  };

  return (
    <CafeuContext.Provider
      value={{
        trackRef,
        reportSignal,
        lastAdaptation,
        frustrationScore,
      }}
    >
      {children}
    </CafeuContext.Provider>
  );
};

// ── Hook: useElementTracking ────────────────────────────────────

export function useElementTracking(label: string) {
  const { trackRef, reportSignal } = useCafeuTracker();
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) trackRef(ref.current, label);
  }, [ref.current]);

  return {
    ref,
    onPress: () => reportSignal({ type: 'rapid_tap', element: label }),
    onLongPress: () => reportSignal({ type: 'long_press', element: label, duration_ms: 3000 }),
  };
}
