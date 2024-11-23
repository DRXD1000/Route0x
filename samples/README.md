## Details on using Route0x for different nuanced routing requirements


| Scenario | Link |
|---------------|----------|
|Get me started | [Snippet](#get-me-started) |
|I have a list of routes and I can give 2 samples per route | [Snippet](#get-me-started) |
|I have a full-on train.csv and test.csv  | [Snippet](#i-have-traincsv-and-testcsv) |
|I have a list of routes but no samples | T.B.A |
|I would like to handle In-domain / adversarial OOS queries  | [Snippet](#i-want-to-handle-id-or-adversarial-oos-queries) |
|I would like to handle Only In-domain queries No OOS  | T.B.A |
|I would like my router robust to typos in queries ? | T.B.A |
|I would like run route0x on a TODS style benchmark dataset to compare performance | T.B.A |


### Get me started:

```python

# ONLY IF YOU WANT TO USE OAI GPT4x Models, This must be set before importing route0x classes
import os
os.putenv("OPENAI_API_KEY", "<your_api_key>")
os.environ["OPENAI_API_KEY"] = "<your_api_key>"

from route0x.route_builder import RouteBuilder, Route, RouteBuilderRequest

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
            llm_name="llama3.1",# gpt4* offers better quality data, API key searched in env, os.getenv("OPENAI_API_KEY")
            enable_synth_data_gen = True,
            enable_id_oos_gen = True,
            max_query_len = 24,
            log_level = "info",
    )

routebuilder.build_routes()

```


### I have train.csv and test.csv

- *Assumes your intention is to get a router out of your train.csv that has atleast 2 samples per route and you want to test the performance of the router on the test.csv you have prepared.* 
- *Use add_additional_invalid_routes only when the TODS is very domain specific formal niche conversations because invalid_routes by default excludes  "chitchat" if you want to allow a more informal chat include the right invalid_routes or set add_additional_invalid_routes to false*

```python

from route0x.route_builder import RouteBuilder, Route, RouteBuilderRequest

routebuilder = RouteBuilder(
            seed = 1234,
            train_path = "<your_train.csv>",
            domain="<your domain description>",
            llm_name="llama3.1",# gpt4* offers better quality data, API key searched in env, os.getenv("OPENAI_API_KEY")
            enable_synth_data_gen = True,
            samples_per_route = 30 # This is default and is more than enough but if you want to experiment and are willing to spend few extra cents you can increase it, best practice it is to increase it in folds of 20s i.e. 50, 70, 90
            max_query_len = 24, # if you have longer queries increase accordingly, but it will have performance penalty
            add_additional_invalid_routes = True # If you have a few additional invalid routes that you want to avoid
            invalid_routes = ["gibberish", "mojibake", "chitchat", "non-english", "profanity"] # Exclude elements as you see fit
            only_gen_dataset = True # enable this to only generate a train and eval set to examine the variety and quality without training.
            log_level = "info",
    )

routebuilder.build_routes()

```


### I want to handle ID or adversarial OOS queries

- *Assumes your intention is to get a router out of your train.csv that has atleast 2 samples per route and you want to test the performance of the router on the test.csv you have prepared.* 


```python

from route0x.route_builder import RouteBuilder, Route, RouteBuilderRequest

routebuilder = RouteBuilder(
            seed = 1234,
            train_path = "<your_train.csv>",
            domain="<your domain description>",
            llm_name="llama3.1",
            enable_synth_data_gen = True,
            enable_id_oos_gen = True, # Include this
            samples_per_route = 30 
            max_query_len = 24, 
            log_level = "info",
    )

routebuilder.build_routes()

```



### FAQs:
- T.B.A

### Knobs and Tips

1. `train/eval/test` csvs expect two columns `text` and `label`.
2. All generated synthetic queries will be in the folder `generated_datasets`
3. `min_samples` signifies the no of samples per route expected for training and defaults to 12.
4. `samples_per_route` signifies how many synthetic samples to be generated per route.
5. route_builder obj method build_params() prints all defaults.
6. route_finder obj method route_params() prints all defaults.
