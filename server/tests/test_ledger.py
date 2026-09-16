import tempfile
import unittest
from pathlib import Path

from server.app.core.ledger import EventType, EvidenceLedger, LedgerEntry


class TestEvidenceLedger(unittest.TestCase):
    def test_append_order_and_chain_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = EvidenceLedger(str(Path(directory) / "ledger.sqlite3"))
            first_id = ledger.append_entry(LedgerEntry.create("wf-test", EventType.WORKFLOW_STARTED))
            ledger.append_entry(LedgerEntry.create("wf-test", EventType.WORKFLOW_COMPLETED, {"ok": True}, first_id))

            entries = ledger.get_workflow_entries("wf-test")

            self.assertEqual(len(entries), 2)
            self.assertEqual(str(entries[1].parent_entry_id), first_id)
            self.assertTrue(ledger.verify_chain("wf-test"))

    def test_tampering_breaks_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = EvidenceLedger(str(Path(directory) / "ledger.sqlite3"))
            ledger.append_entry(LedgerEntry.create("wf-test", EventType.WORKFLOW_STARTED, {"ok": True}))
            ledger.get_workflow_entries("wf-test")

            connection = ledger._connect()
            try:
                connection.execute("UPDATE ledger_entries SET payload_json = ?", ('{"ok":false}',))
                connection.commit()
            finally:
                connection.close()

            self.assertFalse(ledger.verify_chain("wf-test"))


if __name__ == "__main__":
    unittest.main()