from datetime import datetime, timedelta, timezone
from dateutil import parser
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import numpy
import json

from utils import time_to_15min_index, index_to_time, time_to_iso, CITY_TIMEZONES

import random


# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar", 
          "https://www.googleapis.com/auth/calendar.events", 
          "https://www.googleapis.com/auth/calendar.readonly", 
          "https://www.googleapis.com/auth/calendar.calendarlist", 
          "https://www.googleapis.com/auth/calendar.calendarlist.readonly"]

token_file = "calendar_interface/token.json"
creds_file = "calendar_interface/credentials.json"

with open('calendar_details.json', 'r', encoding='utf-8') as f:
    event_information = json.load(f)

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


        if start_dt != "Invalid date format":
            if start_dt is None or start_dt < next_day_start or start_dt >= next_day_end:
                continue
            
            day_offset = (start_dt.date() - next_day_start.date()).days
            time_block_start = time_to_15min_index(datetime.fromisoformat(start_time_str))
            time_block_end = time_to_15min_index(datetime.fromisoformat(end_time_str))

            if 0 <= day_offset < days and 0 <= time_block_start < 96 and 0 <= time_block_end < 96:
                schedule[time_block_start:time_block_end+1, day_offset] = task_id


def fetch_data(schedule):

  days = schedule.shape[1]

  creds = None
  if os.path.exists(token_file):
    creds = Credentials.from_authorized_user_file(token_file, SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          creds_file, SCOPES
      )
      creds = flow.run_local_server(port=0)
    with open(token_file, "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)
    cal_ids, cal_dict, cal_names = fetch_calendars(service)
    for i, idx in enumerate(cal_ids):
      fetch_events(service, idx, days, schedule, cal_dict)
      
    return schedule, cal_dict, cal_names


  except HttpError as error:
    print(f"An error occurred: {error}")

def write_to_calendar(prev_schedule, schedule, cal_dict:dict, cal_names:list):

  #Safety block to allow for new credentials
  creds = None
  if os.path.exists(token_file):
    creds = Credentials.from_authorized_user_file(token_file, SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          creds_file, SCOPES
      )
      creds = flow.run_local_server(port=0)
    with open(token_file, "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)
  except HttpError as error:
    print(f"An error occurred: {error}")

  
  #Only execute new events
  schedule_mask = numpy.logical_not(numpy.logical_and(prev_schedule, schedule))

  for day in range(schedule.shape[1]):
    schedule_readable = schedule_to_idx(numpy.multiply(schedule_mask, schedule)[:, day])
    today_str = datetime.now().strftime("%Y-%m-%d")
    current_day = datetime.strptime(today_str, "%Y-%m-%d") + timedelta(days=day+1)
    current_day_str = current_day.strftime("%Y-%m-%d")
    schedule_w_idx = list(map(lambda x: [time_to_iso("Zurich", index_to_time(x[1]), current_day_str, False), time_to_iso("Zurich", index_to_time(x[2]), current_day_str, False), list(cal_dict.keys())[list(cal_dict.values()).index(x[0])], 
                                         cal_names[int(x[0])-1], index_to_time(x[1]), index_to_time(x[2])], schedule_readable))
    
    for event_raw in schedule_w_idx:
      this_event = {
          'summary': f"{random.choice(event_information[event_raw[3]]['labels'])}",
          'start':{
             'dateTime': f'{event_raw[0]}',
             'timeZone': CITY_TIMEZONES['Zurich']
          },
          'location': f"{event_information[event_raw[3]].get('location')}",
          'end':{
             'dateTime': f'{event_raw[1]}',
             'timeZone': CITY_TIMEZONES['Zurich']
          },
          'reminders': {
              'useDefault': False,
              'overrides': [
                {'method': 'popup', 'minutes': 30},
              ],
          },
       }

      event = service.events().insert(calendarId=f'{event_raw[2]}', body = this_event).execute()

def schedule_to_idx(arr):
  result = []
  i = 0
  while i < len(arr):
    if arr[i] > 0:
        value = arr[i]
        start = i
        while i + 1 < len(arr) and arr[i + 1] == value:
            i += 1
        end = i
        result.append([int(value), start, end+1])
    i += 1

  return result