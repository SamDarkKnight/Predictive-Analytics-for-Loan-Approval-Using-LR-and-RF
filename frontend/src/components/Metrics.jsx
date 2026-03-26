export default function Metrics({data}) {
  return (
    <div className="grid">
      <div>Accuracy: {(data.accuracy*100).toFixed(1)}%</div>
      <div>ROC-AUC: {data.roc_auc.toFixed(3)}</div>
      <div>Approved: {data.class_dist.Approved}</div>
      <div>Rejected: {data.class_dist.Rejected}</div>
    </div>
  );
}