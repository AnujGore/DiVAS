from datetime import datetime, timedelta, timezone
from dateutil import parser
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar", 
          "https://www.googleapis.com/auth/calendar.events", 
          "https://www.googleapis.com/auth/calendar.readonly", 
          "https://www.googleapis.com/auth/calendar.calendarlist", 
          "https://www.googleapis.com/auth/calendar.calendarlist.readonly"]


def format_datetime(input_string):
    try:
        if input_string is not None:
          dt = parser.parse(input_string)
          # If time is not included, default to just showing the date
          if dt.time() == dt.min.time():
              return dt.strftime("%B %d, %Y")
          else:
              # Format with timezone if available
              if dt.tzinfo:
                  readable = dt.strftime("%B %d, %Y at %I:%M%p")
                  readable = readable[:-2] + ":" + readable[-2:]  # Make timezone like +01:00
              else:
                  readable = dt.strftime("%B %d, %Y at %I:%M %p")
              return readable
        else:
           return 
    except ValueError:
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

  return calendar_ids, calendar_names

def fetch_events(service, id, days):
    days = 7

    # Calculate next day's start and end time
    now = datetime.now(tz = timezone.utc)
    next_day_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    next_day_end = next_day_start + timedelta(days=days)

    # Format times in RFC3339 (ISO 8601) with timezone info
    time_min = next_day_start.isoformat()
    time_max = next_day_end.isoformat()

    # Call the Google Calendar API
    events_result = (
        service.events()
        .list(
            calendarId=id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    
    events = events_result.get("items", [])

    if not events:
      print("No upcoming events found.")
      return

    # Prints the start and name of the next 10 events
    for event in events:
      print(format_datetime(event["start"].get("dateTime")), event["summary"])



def fetch_data():
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
    cal_ids, cal_names = fetch_calendars(service)
    for i, idx in enumerate(cal_ids):
      print(cal_names[i])
      fetch_events(service, idx, 7)
      print()

  except HttpError as error:
    print(f"An error occurred: {error}")


if __name__ == "__main__":
  fetch_data()
