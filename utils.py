from datetime import datetime, timedelta

def summarize_day_array(day_array, start_hour=9):
    block_length = 15  # minutes
    start_time = datetime(2025, 4, 17, start_hour, 0)

    current_task = day_array[0]
    start_block = 0
    schedule = []

    for i in range(1, len(day_array)):
        if day_array[i] != current_task:
            # Time to finalize current block
            end_block = i
            schedule.append((current_task, start_block, end_block))
            start_block = i
            current_task = day_array[i]
    
    # Add the last segment
    schedule.append((current_task, start_block, len(day_array)))

    # Now print it nicely
    for task_id, start_idx, end_idx in schedule:
        task_label = f"Task {int(task_id)}" if task_id > 0 else "Lunch Time"
        start_time_str = (start_time + timedelta(minutes=start_idx * block_length)).strftime("%H:%M")
        end_time_str = (start_time + timedelta(minutes=end_idx * block_length)).strftime("%H:%M")
        duration = (end_idx - start_idx) * block_length
        print(f"- {task_label}: {start_time_str} to {end_time_str} ({duration} mins)")