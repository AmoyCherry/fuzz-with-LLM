## SyzLLM Server

## Integration

### modify syzkaller to avoid panic

#### panic: both fault injection and rerun are enabled for the same call

- Assign 0 to `Call.CallProps.Rerun` for each call recieved from SyzLLM

- Commenting the fatal in executor.cc.



## VM

#### fatal: too much open file

- Commenting the panic for syz_usbip_server_init in common_linux.h.

