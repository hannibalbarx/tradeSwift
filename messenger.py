from boto.sqs.message import RawMessage
import boto.sqs

import sys

conn = boto.sqs.connect_to_region("us-east-1")
tasks_q = conn.get_queue("TasksQueue")
results_q = conn.get_queue("ResultsQueue")

while True:
		cur_message= tasks_q.read()
		if cur_message:
			print cur_message.get_body()
			sys.stdout.flush()
			input=raw_input()
			while input!="done":
				raw_message = RawMessage()
				raw_message.set_body(input)
				results_q.write(raw_message)
				input=raw_input()
			tasks_q.delete_message(cur_message)
			
			
