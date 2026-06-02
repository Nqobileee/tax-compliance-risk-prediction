import "./RiskGauge.css";

interface RiskGaugeProps {
  score: number;
  tier: "Low" | "Medium" | "High";
}

export function RiskGauge({ score, tier }: RiskGaugeProps) {
  const pct = Math.round(score * 100);
  const rotation = -90 + score * 180;

  const tierClass =
    tier === "High" ? "gauge-high" : tier === "Medium" ? "gauge-medium" : "gauge-low";

  return (
    <div className={`risk-gauge ${tierClass}`}>
      <div className="gauge-arc">
        <div className="gauge-fill" style={{ transform: `rotate(${rotation}deg)` }} />
        <div className="gauge-center">
          <span className="gauge-value">{pct}</span>
          <span className="gauge-unit">/ 100</span>
          <span className="gauge-tier">{tier} Risk</span>
        </div>
      </div>
      <div className="gauge-glow" />
    </div>
  );
}
