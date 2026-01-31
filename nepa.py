for pixel_values in loader: # x, [B, H, W, C]
input_embed = f(pixel_values) # z, [B, T, D]
pred_embed = h(input_embed) # z_hat, [B, T, D]

loss = D(input_embed, pred_embed) # loss
loss.backward() # back-propagate

update(f.param, h.param) # update parameters

def D(z, z_hat):
target = z.detach() # stop gradient
pred = z_hat[:, 0:T-1, :] # shift, [B, T-1, D]
target = target[:, 1:T, :] # shift, [B, T-1, D]
# Use any suitable distance metric.

pred = normalize(pred, axis=-1) # l2-norm
target = normalize(target, axis=-1) # l2-norm
return -(pred * target).sum(dim=-1).mean()