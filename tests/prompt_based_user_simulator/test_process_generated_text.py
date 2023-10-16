from convlab2.nlg.generative_models.user_simulator_generative_model import (
    UserSimulatorGenerativeModel)


def test_processing_generated_text_with_multiple_user():

    model = UserSimulatorGenerativeModel
    generated_text = "CUSTOMER: I want to find a CUSTOMER: cheap hotel"
    processed_text = model._process_generated_text(
        generated_text=generated_text)
    assert processed_text == 'I want to find a cheap hotel'


def test_processing_generated_text_with_pretext():

    model = UserSimulatorGenerativeModel
    generated_text = "usergoal something CUSTOMER: I want to find a cheap hotel"  # noqa
    processed_text = model._process_generated_text(
        generated_text=generated_text)
    assert processed_text == 'I want to find a cheap hotel'


def test_processing_generated_text_with_assistant():

    model = UserSimulatorGenerativeModel
    generated_text = "CUSTOMER: I want to find a cheap hotel. ASSISTANT: ok"
    processed_text = model._process_generated_text(
        generated_text=generated_text)
    assert processed_text == 'I want to find a cheap hotel.'


def test_processing_generated_text_that_is_too_long():

    model = UserSimulatorGenerativeModel
    generated_text = ' '.join(
        ["CUSTOMER: I want to find a cheap hotel. ASSISTANT: ok"] * 20)
    processed_text = model._process_generated_text(
        generated_text=generated_text)
    assert processed_text == 'I want to find a cheap hotel.'
