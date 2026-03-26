import React, { useState } from "react";

function sigmoid(x){ return 1/(1+Math.exp(-x)); }

export default function Simulator({ data }) {

  const [inc,setInc] = useState(60000);
  const [coin,setCoin] = useState(30000);
  const [cs,setCs] = useState(680);
  const [loan,setLoan] = useState(200000);
  const [term,setTerm] = useState(60);

  function predict(){
    const total = inc+coin;
    const dti = loan/(total+1);
    const monthly = loan/term;
    const cat = cs<600?0:cs<700?1:cs<800?2:3;

    const feats = [inc,coin,cs,loan,term,total,dti,monthly,cat];

    const scaled = feats.map((v,i)=>
      (v-data.scaler_mean[i])/data.scaler_scale[i]
    );

    let z = data.lr_intercept;
    scaled.forEach((v,i)=> z += v*data.lr_coef[i]);

    return sigmoid(z);
  }

  const prob = Math.round(predict()*100);

  return (
    <div className="card">
      <h2>What-if Simulator</h2>

      <input type="range" min="20000" max="300000" value={inc} onChange={e=>setInc(+e.target.value)} />
      <p>Income: ₹{inc}</p>

      <input type="range" min="300" max="900" value={cs} onChange={e=>setCs(+e.target.value)} />
      <p>Credit Score: {cs}</p>

      <h3>Approval Probability: {prob}%</h3>
    </div>
  );
}