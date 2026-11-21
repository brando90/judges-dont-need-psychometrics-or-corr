E = {
'when_w1': ('D', r"\text{excess Brier}=\frac1n\sum_i\big(a+(b-1)t_i\big)^2,\qquad \text{excess CE}=\frac1n\sum_i \mathrm{KL}\big(\mathrm{Bern}(t_i)\,\Vert\,\mathrm{Bern}(a+bt_i)\big)"),
'when_w2': ('D', r"p=(s,\ 1-2s,\ s),\quad s\in[0,\tfrac12],\qquad m=1,\quad \mathrm{Var}=2s,\quad F=(s,\ 1-s)"),
'when_w2b': ('D', r"\mathrm{RPS}_{\mathrm{unnorm}}\big(s=0,\ s=\tfrac12\big)=\big(0-\tfrac12\big)^2+\big(1-\tfrac12\big)^2=\tfrac12"),
'when_w3a': ('D', r"\mathrm{KL}(p^\ast\Vert q_\eta)=\mathrm{const}-\eta\,t+A(\eta),\qquad \frac{d}{d\eta}:\ -t+A'(\eta)=0\ \Longrightarrow\ m(\eta)=t"),
'when_w3': ('D', r"\mathrm{KL}\big(q_{\eta(t)}\Vert q_{\eta(m)}\big)\approx\frac{(m-t)^2}{2\,V(t)}"),
'when_w4': ('D', r"\Theta_\rho=\{\theta:\ M(\theta)=a+b\,M(\theta^\ast),\ b>0\}\ \supseteq\ \{\theta^\ast\}"),
'when_w5': ('D', r"\rho(w)=\frac{w^\top c}{\sqrt{w^\top\Sigma w}\,\sigma_t}\ \le\ \sqrt{\frac{c^\top\Sigma^{-1}c}{\sigma_t^2}},\qquad \text{equality iff } w=\beta\,\Sigma^{-1}c\ \ (\beta>0)"),
'when_w7': ('D', r"\rho(m,\hat t)=\frac{\rho(m,t)}{\sqrt{1+\bar\sigma^2/(k\,s_t^2)}},\qquad \hat t_i=t_i+\varepsilon_i,\ \ \mathrm{Var}\,\varepsilon_i=\sigma_i^2/k"),
'when_w7b': ('D', r"\mathbb E[\text{excess KL of the CE head}]\approx\frac{P}{2N}\ \ (N\gg P),\qquad N^\ast\approx\frac{P}{2B}"),
}
