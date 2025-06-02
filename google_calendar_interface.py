from datetime import datetime, timedelta, timezone
from dateutil import parser
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from utils import time_to_15min_index


# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar", 
          "https://www.googleapis.com/auth/calendar.events", 
          "https://www.googleapis.com/auth/calendar.readonly", 
          "https://www.googleapis.com/auth/calendar.calendarlist", 
          "https://www.googleapis.com/auth/calendar.calendarlist.readonly"]


def format_datetime(input_string):
    try:
      dt = parser.parse(input_string)
      return dt
    except Exception as e:
        return "Invalid date format"


def fetch_calendars(service):
  """
  Returns: Calendar IDs, Calendar Names
  """
  calendar_ids = []
  calendar_names = []
  all_calendars = service.calendarList().list()
  all_calendars = all_calendars.execute()
  for cal in all_calendars['items']:
    if cal['summary'] not in ['Week Numbers', 'Work']:
      calendar_names.append(cal['summary'])
      calendar_ids.append(cal['id'])

  calendar_dict = {idx: i+1 for i, idx in enumerate(calendar_ids)}

  return calendar_ids, calendar_dict, calendar_names

def fetch_events(service, calendar_id, days, schedule, task_dict):
    now = datetime.now(tz=timezone.utc)
    next_day_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    next_day_end = next_day_start + timedelta(days=days)

    time_min = next_day_start.isoformat()
    time_max = next_day_end.isoformat()

    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = events_result.get("items", [])

    task_id = task_dict.get(calendar_id, 10)  # Default to 10

    for event in events:
        start_time_str = event["start"].get("dateTime")
        end_time_str = event["end"].get("dateTime")
        start_dt = format_datetime(start_time_str)
        end_dt = format_datetime(end_time_str)

        if start_dt != "Invalid date format":
            if start_dt is None or start_dt < next_day_start or start_dt >= next_day_end:
                continue
            

            day_offset = (start_dt.date() - next_day_start.date()).days
            time_block_start = time_to_15min_index(start_dt.strftime("%H:%M"))
            time_block_end = time_to_15min_index(end_dt.strftime("%H:%M"))

            if 0 <= day_offset < days and 0 <= time_block_start < 96 and 0 <= time_block_end < 96:
                schedule[time_block_start:time_block_end+1, day_offset] = task_id


def fetch_data(schedule):

  days = schedule.shape[1]

  creds = None
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)
    cal_ids, cal_dict, _ = fetch_calendars(service)
    for i, idx in enumerate(cal_ids):
      fetch_events(service, idx, days, schedule, cal_dict)
      
    return schedule


  except HttpError as error:
    print(f"An error occurred: {error}")
