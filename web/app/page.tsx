import { LiveDemo } from "@/components/demo/LiveDemo";
import { readConfigMeta, readMapPoints } from "@/lib/server/upstream";
import type { MapPointsFile, Meta } from "@/lib/api";

export default function LiveDemoPage() {
  let mapPoints: MapPointsFile = {
    n_total: 0,
    n_projected: 0,
    n_saved: 0,
    method: "UMAP",
    coordinates: [],
  };
  let initialMeta: Meta | null = null;

  try {
    mapPoints = readMapPoints();
  } catch {
    // The demo still works without the semantic map.
  }

  try {
    initialMeta = readConfigMeta();
  } catch {
    initialMeta = null;
  }

  return (
    <LiveDemo
      mapPoints={mapPoints.coordinates}
      initialMeta={initialMeta}
    />
  );
}
