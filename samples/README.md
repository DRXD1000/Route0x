## Details on using Route0x different nuanced routing requirements


FAQs:
1. `train/eval/test` csvs expect two columns `text` and `label`.
2. All generated synthetic queries will be in the folder `generated_datasets`
3. `min_samples` signifies the no of samples per route expected for training and defaults to 12.
4. `samples_per_route` signifies how many synthetic samples to be generated per route.
4. routebuilder obj method show_defaults() prints all defaults.



| Scenario | Link |
|---------------|----------|
|Get me started | [Snippet](#get-me-started) |
|I have a list of routes and I can give 2 samples per route | T.B.A |
|I have a list of routes but no samples | T.B.A |
|I have a full-on train.csv and test.csv  | T.B.A |
|I would like to handle In-domain / adversarial OOS queries  | T.B.A |
|I would like my router robust to typos in queries | T.B.A |
|I would like run route0x on a TODS style benchmark dataset to compare performance | T.B.A |


### Get me started:

```python

from route0x import RouteBuilder, Route, RouteBuilderRequest

routes = [
  Route("alarm/cancel_alarm", ['delete my alarms please', 'cancel oven alarm']),
  Route("alarm/modify_alarm", ['Change alarm setting to 5am.', 'change my 5am alarm to 6am please']),
  Route("alarm/set_alarm", ['Set alarm to go off every 30 seconds for 20 minutes', 'Set alarm every 3 minutes for 20 minutes.']),
  Route("alarm/show_alarms", ['When is my next alarm for', 'when is my alarm going to go off']),
  Route("alarm/snooze_alarm", ['Snooze the alarm for 20 mins.', 'snooze all the alarms']),
  Route("alarm/time_left_on_alarm", ['How much time is left until my alarm rings?', 'How much time do I have left on my alarm?']),
  Route("reminder/cancel_reminder", ['delete my two most recent reminders', 'Clear all reminders for this weekend']),
  Route("reminder/set_reminder", ['Remind me to look for a dress for the wedding on Friday', 'Remind me to get dog food at 4:30 pm']),
  Route("reminder/show_reminders", ['remind me of when I need to leave for my flight on Friday', 'Show my Reminders every 30 minutes until I swipe them as done']),
  Route("weather/checkSunrise", ['What time does the sun come up tomorrow', 'what time is sunrise tomorrow']),
  Route("weather/checkSunset", ['What time is sunset on friday', 'When does the sun set today?']),
  Route("weather/find", ['Whats the high for the next 3 days?', 'what is the temperature high and low for today'])
]

build_request = RouteBuilderRequest(routes)


routebuilder = RouteBuilder(
            seed = 1234,
            build_request = build_request,
            domain="personal assistant",
            llm_name="llama3.1",
            enable_synth_data_gen = True,
            enable_id_oos_gen = True,
    )
```

### Knobs