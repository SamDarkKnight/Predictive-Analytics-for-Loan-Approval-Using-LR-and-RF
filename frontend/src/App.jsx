import React, { useEffect, useState } from "react";
import Simulator from "./components/Simulator";
import Metrics from "./components/Metrics";
import Segments from "./components/Segments";
import ConfusionMatrix from "./components/ConfusionMatrix";
import data from "./data/simulator_data.json";

export default function App() {
  const [DATA, setData] = useState(null);

  useEffect(() => {
    setData(data);
  }, []);

  if (!DATA) return <div>Loading...</div>;

  return (
    <div className="app">
      <h1>⚡ Loan Risk Intelligence</h1>

      <Metrics data={DATA} />

      <Simulator data={DATA} />

      <Segments data={DATA} />

      <ConfusionMatrix data={DATA} />
    </div>
  );
}