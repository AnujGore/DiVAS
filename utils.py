from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo 
from typing import List

CITY_TIMEZONES = {
    "Zurich": "Europe/Zurich",
    "Los Angeles": "America/Los_Angeles",
}

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


def time_to_15min_index(dt):
    """
    Convert a timezone-aware datetime to a 15-minute index
    using the local time (with its timezone offset included).
    """
    local_dt = dt.astimezone()  # converts to local timezone
    hours = local_dt.hour
    minutes = local_dt.minute
    index = hours * 4 + (minutes // 15)
    return index

def index_to_time(idx):
    hours = idx // 4
    minutes = (idx % 4) * 15
    return f"{hours:02}:{minutes:02}"


def time_to_iso(city: str, time: str, date_str: str = None, input_is_utc: bool = False) -> List[str]:
    """
    Convert a list of times to ISO 8601 strings with correct timezone.

    :param city: City name, e.g., "Zurich"
    :param times: List of times in HH:MM format
    :param date_str: Date in YYYY-MM-DD format
    :param input_is_utc: If True, input times are UTC and will be converted to local time
    :return: List of ISO 8601 strings with timezone
    """
    if city not in CITY_TIMEZONES:
        raise ValueError(f"Timezone for city '{city}' not defined.")
    
    tz = ZoneInfo(CITY_TIMEZONES[city])

    if date_str is None:
        date_str = date.today().isoformat()

    dt_naive = datetime.strptime(f"{date_str} {time}", "%Y-%m-%d %H:%M")
    
    if input_is_utc:
        # Input is UTC, convert to local timezone
        dt_utc = dt_naive.replace(tzinfo=ZoneInfo("UTC"))
        dt_local = dt_utc.astimezone(tz)
    else:
        # Input is local time
        dt_local = dt_naive.replace(tzinfo=tz)

    return dt_local.isoformat()