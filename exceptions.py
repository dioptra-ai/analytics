
class HandledExceptions(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

class ClientException(HandledExceptions):
    def __init__(self, message, status_code=400):
        HandledExceptions.__init__(
            self, status_code=status_code, message=message)

class ServerException(HandledExceptions):
    def __init__(self, message, status_code=500):
        HandledExceptions.__init__(
            self, status_code=status_code, message=message)

class IllegalArgumentError(ClientException):
    def __init__(self, error):
        ClientException.__init__(
            self, message=f'Invalid request: {error}')

class EmbeddingsShapeMismatchError(ClientException):
    def __init__(self, reference_dimension, current_dimension):
        ClientException.__init__(
            self,
            message=f'Embeddings and reference embeddings dimensions didn\'t match: {current_dimension} vs {reference_dimension}')

class ReferenceNoDataError(ClientException):
    def __init__(self):
        ClientException.__init__(
            self, message='The reference data is empty')

class CurrentNoDataError(ClientException):
    def __init__(self):
        ClientException.__init__(
            self, message='The data is empty')

class NotEnoughDataError(ClientException):
    def __init__(self, limit):
        ClientException.__init__(
            self, message=f'Not enough data: at least {limit} vectors are required for this analysis.')

class NotImplementedError(ServerException):
    def __init__(self, message):
        ServerException.__init__(
            self, status_code=501, message=message)