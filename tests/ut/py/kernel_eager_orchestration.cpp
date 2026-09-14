/* Minimal TMR eager smoke orchestration. */
#include "orchestration_api.h"

extern "C" void kernel_eager_orchestration(const ChipTaskArgs &args) {
    CoreTaskArgs task;
    task.add_input(args.tensor(0).ref());
    task.add_output(args.tensor(1).ref());
    task.add_scalar(args.scalar(0));
    rt_submit_aiv_task(0, task);
}
