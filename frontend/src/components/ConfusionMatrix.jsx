export default function ConfusionMatrix({data}) {
  const [[tn,fp],[fn,tp]] = data.confusion_matrix;

  return (
    <div>
      <h2>Confusion Matrix</h2>
      <p>TN: {tn} | FP: {fp}</p>
      <p>FN: {fn} | TP: {tp}</p>
    </div>
  );
}