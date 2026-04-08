from emergency_dispatch.tasks import EasyDispatchTask, MediumDispatchTask, HardDispatchTask
import statistics

results = {}
for TaskCls in [EasyDispatchTask, MediumDispatchTask, HardDispatchTask]:
    scores = []
    for seed in [7, 42, 99, 123, 256]:
        task = TaskCls()
        env = task.create_env(seed=seed)
        grader = task.create_grader()
        state = env.reset()
        done = False
        while not done:
            action = env.heuristic_action()
            state, reward, done, _ = env.step(action)
        score = grader.grade(state, task_name=task.name).final_score
        scores.append(score)
    mean = round(statistics.mean(scores), 4)
    stdev = round(statistics.stdev(scores), 4)
    results[task.name] = {"mean": mean, "stdev": stdev}
    print(f"{task.name}: mean={mean} stdev={stdev}", flush=True)

print("DONE", results)
