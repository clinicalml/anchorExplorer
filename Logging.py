import time

class LogElement:
	"""docstring for LogElement"""
	def __init__(self, elemType, note='', details={}):
		self.type = elemType
		self.note = note
		self.details = {}
		self.timestamp = time.time()
