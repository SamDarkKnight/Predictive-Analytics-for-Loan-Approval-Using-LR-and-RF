export default function Segments({data}) {
  return (
    <div>
      <h2>Risk Segments</h2>
      {Object.entries(data.segment_counts).map(([k,v])=>(
        <div key={k}>
          {k}: {v} users | {(data.segment_approval_rates[k]*100).toFixed(0)}% approved
        </div>
      ))}
    </div>
  );
}