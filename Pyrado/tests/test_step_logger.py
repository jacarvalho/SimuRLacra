import csv
import pickle
import pytest
import unittest.mock as mock

import pyrado.logger.step as uut


@pytest.mark.logger
def test_first_step():
    ap = mock.Mock(uut.StepLogPrinter)
    logger = uut.StepLogger()
    logger.printers.append(ap)

    # Test first step
    logger.add_value('Dummy', 1)
    logger.record_step()
    ap.print_values.assert_called_once_with(mock.ANY, mock.ANY, True)
    ap.reset_mock()

    # Test second step
    logger.add_value('Dummy', 2)
    logger.record_step()
    ap.print_values.assert_called_once_with(mock.ANY, mock.ANY, False)
    ap.reset_mock()

    # Test third step
    logger.add_value('Dummy', 3)
    logger.record_step()
    ap.print_values.assert_called_once_with(mock.ANY, mock.ANY, False)
    ap.reset_mock()


@pytest.mark.logger
def test_values():
    ap = mock.Mock(uut.StepLogPrinter)
    logger = uut.StepLogger()
    logger.printers.append(ap)

    # Test one value combi
    logger.add_value('Value1', 1)
    logger.add_value('Value2', 20)
    logger.record_step()
    ap.print_values.assert_called_once_with({'Value1': 1, 'Value2': 20}, mock.ANY, mock.ANY)
    ap.reset_mock()

    # And another
    logger.add_value('Value1', 12)
    logger.add_value('Value2', -6.7)
    logger.record_step()
    ap.print_values.assert_called_once_with({'Value1': 12, 'Value2': -6.7}, mock.ANY, mock.ANY)
    ap.reset_mock()

    # Only update value1 - value2 should still be there
    logger.add_value('Value1', 14)
    logger.record_step()
    ap.print_values.assert_called_once_with({'Value1': 14, 'Value2': -6.7}, mock.ANY, mock.ANY)
    ap.reset_mock()


@pytest.mark.logger
def test_consistent_key_order():
    ap = mock.Mock(uut.StepLogPrinter)
    logger = uut.StepLogger()
    logger.printers.append(ap)

    # Add Value1 first
    logger.add_value('Value1', 1)
    logger.add_value('Value2', 20)
    logger.record_step()
    ap.print_values.assert_called_once_with(mock.ANY, ['Value1', 'Value2'], mock.ANY)
    ap.reset_mock()

    # Now add value2 first
    logger.add_value('Value2', -6.7)
    logger.add_value('Value1', 12)
    logger.record_step()
    ap.print_values.assert_called_once_with(mock.ANY, ['Value1', 'Value2'], mock.ANY)
    ap.reset_mock()


@pytest.mark.logger
def test_empty_step_skip():
    ap = mock.Mock(uut.StepLogPrinter)
    logger = uut.StepLogger()
    logger.printers.append(ap)

    # Record a step
    logger.add_value('Dummy', 20)
    logger.record_step()
    ap.print_values.assert_called_once()
    ap.reset_mock()

    # Call record step without adding a value - should not call printer
    logger.record_step()
    ap.print_values.assert_not_called()
    ap.reset_mock()

    # Add a value again
    logger.add_value('Dummy', 24)
    logger.record_step()
    ap.print_values.assert_called_once()
    ap.reset_mock()


@pytest.mark.logger
def test_late_new_key_error():
    ap = mock.Mock(uut.StepLogPrinter)
    logger = uut.StepLogger()
    logger.printers.append(ap)

    # Record a step
    logger.add_value('Value1', 1)
    logger.add_value('Value2', 20)
    logger.record_step()

    # Try to add an unknown key
    with pytest.raises(KeyError):
        logger.add_value('Unknown', 42)


@pytest.mark.logger
def test_prefix():
    ap = mock.Mock(uut.StepLogPrinter)
    logger = uut.StepLogger()
    logger.printers.append(ap)

    # Record plain value
    logger.add_value('Value0', 1)

    # Record prefixed value manually
    logger.push_prefix('Prefix1_')
    logger.add_value('Value1', 2)
    logger.pop_prefix()

    # Record prefixed value with contextmanager
    with logger.prefix('Prefix2_'):
        logger.add_value('Value2', 2)

    # Assert key names are correct
    logger.record_step()
    ap.print_values.assert_called_once_with(mock.ANY, ['Value0', 'Prefix1_Value1', 'Prefix2_Value2'], mock.ANY)


@pytest.mark.logger
def test_csv_logger_serializer(tmpdir):
    outfile = tmpdir/'testout.csv'

    # Create csv logger
    cp = uut.CSVPrinter(outfile)
    logger = uut.StepLogger()
    logger.printers.append(cp)

    # Log some values
    logger.add_value('Value1', 10)
    logger.add_value('Value2', 20)
    logger.record_step()

    # Ser/deser
    logger_reser = pickle.loads(pickle.dumps(logger, pickle.HIGHEST_PROTOCOL))
    # Log values with new logger
    logger_reser.add_value('Value1', 100)
    logger_reser.add_value('Value2', 200)
    logger_reser.record_step()

    # This should have properly appended to the csv file
    with outfile.open() as outfilehandle:
        rows = list(csv.DictReader(outfilehandle))

    assert rows[0]['Value1'] == '10'
    assert rows[0]['Value2'] == '20'
    assert rows[1]['Value1'] == '100'
    assert rows[1]['Value2'] == '200'
