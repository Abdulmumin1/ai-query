"""Tests for Storage backends."""

import pytest
from ai_query.agents.storage import Storage, MemoryStorage, SQLiteStorage


class TestStorageProtocol:
    """Tests for Storage protocol compliance."""

    def test_memory_storage_implements_protocol(self):
        """MemoryStorage should implement Storage protocol."""
        storage = MemoryStorage()
        assert isinstance(storage, Storage)

    def test_sqlite_storage_implements_protocol(self, tmp_path):
        """SQLiteStorage should implement Storage protocol."""
        storage = SQLiteStorage(str(tmp_path / "test.db"))
        assert isinstance(storage, Storage)
        storage.close()


class TestMemoryStorage:
    """Tests for MemoryStorage."""

    @pytest.fixture
    def storage(self):
        """Create a fresh MemoryStorage instance."""
        return MemoryStorage()

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, storage):
        """get() should return None for missing keys."""
        result = await storage.get("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, storage):
        """set() should store values retrievable by get()."""
        await storage.set("key1", "value1")
        result = await storage.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_set_overwrites(self, storage):
        """set() should overwrite existing values."""
        await storage.set("key1", "original")
        await storage.set("key1", "updated")
        result = await storage.get("key1")
        assert result == "updated"

    @pytest.mark.asyncio
    async def test_set_complex_types(self, storage):
        """set() should handle dicts, lists, and nested structures."""
        data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "nested": {"a": {"b": "c"}}
        }
        await storage.set("complex", data)
        result = await storage.get("complex")
        assert result == data

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        """delete() should remove keys."""
        await storage.set("key1", "value1")
        await storage.delete("key1")
        result = await storage.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, storage):
        """delete() should not raise for missing keys."""
        await storage.delete("missing")

    @pytest.mark.asyncio
    async def test_keys_empty(self, storage):
        """keys() should return empty list when no keys."""
        result = await storage.keys()
        assert result == []

    @pytest.mark.asyncio
    async def test_keys_all(self, storage):
        """keys() should return all keys."""
        await storage.set("a", 1)
        await storage.set("b", 2)
        await storage.set("c", 3)
        result = await storage.keys()
        assert sorted(result) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_keys_with_prefix(self, storage):
        """keys() should filter by prefix."""
        await storage.set("user:1:name", "Alice")
        await storage.set("user:1:email", "alice@example.com")
        await storage.set("user:2:name", "Bob")
        await storage.set("settings:theme", "dark")

        user1_keys = await storage.keys("user:1:")
        assert sorted(user1_keys) == ["user:1:email", "user:1:name"]

        user_keys = await storage.keys("user:")
        assert len(user_keys) == 3

        settings_keys = await storage.keys("settings:")
        assert settings_keys == ["settings:theme"]

    def test_clear(self, storage):
        """clear() should remove all data."""
        storage._data["key1"] = "value1"
        storage._data["key2"] = "value2"
        storage.clear()
        assert storage._data == {}


class TestSQLiteStorage:
    """Tests for SQLiteStorage."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a fresh SQLiteStorage instance."""
        storage = SQLiteStorage(str(tmp_path / "test.db"))
        yield storage
        storage.close()

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, storage):
        """get() should return None for missing keys."""
        result = await storage.get("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, storage):
        """set() should store values retrievable by get()."""
        await storage.set("key1", "value1")
        result = await storage.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_set_overwrites(self, storage):
        """set() should overwrite existing values."""
        await storage.set("key1", "original")
        await storage.set("key1", "updated")
        result = await storage.get("key1")
        assert result == "updated"

    @pytest.mark.asyncio
    async def test_set_complex_types(self, storage):
        """set() should handle dicts, lists, and nested structures."""
        data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "nested": {"a": {"b": "c"}}
        }
        await storage.set("complex", data)
        result = await storage.get("complex")
        assert result == data

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        """delete() should remove keys."""
        await storage.set("key1", "value1")
        await storage.delete("key1")
        result = await storage.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, storage):
        """delete() should not raise for missing keys."""
        await storage.delete("missing")

    @pytest.mark.asyncio
    async def test_keys_empty(self, storage):
        """keys() should return empty list when no keys."""
        result = await storage.keys()
        assert result == []

    @pytest.mark.asyncio
    async def test_keys_all(self, storage):
        """keys() should return all keys."""
        await storage.set("a", 1)
        await storage.set("b", 2)
        await storage.set("c", 3)
        result = await storage.keys()
        assert sorted(result) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_keys_with_prefix(self, storage):
        """keys() should filter by prefix."""
        await storage.set("user:1:name", "Alice")
        await storage.set("user:1:email", "alice@example.com")
        await storage.set("user:2:name", "Bob")
        await storage.set("settings:theme", "dark")

        user1_keys = await storage.keys("user:1:")
        assert sorted(user1_keys) == ["user:1:email", "user:1:name"]

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        """Data should persist across instances."""
        db_path = str(tmp_path / "persist.db")

        storage1 = SQLiteStorage(db_path)
        await storage1.set("persistent", {"value": 123})
        storage1.close()

        storage2 = SQLiteStorage(db_path)
        result = await storage2.get("persistent")
        assert result == {"value": 123}
        storage2.close()

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path):
        """SQLiteStorage should create parent directories if needed."""
        db_path = str(tmp_path / "nested" / "dir" / "test.db")
        storage = SQLiteStorage(db_path)
        await storage.set("key", "value")
        result = await storage.get("key")
        assert result == "value"
        storage.close()


class TestStorageIsolation:
    """Tests for storage instance isolation."""

    @pytest.mark.asyncio
    async def test_memory_storage_instances_isolated(self):
        """Separate MemoryStorage instances should not share data."""
        storage1 = MemoryStorage()
        storage2 = MemoryStorage()

        await storage1.set("key", "value1")
        await storage2.set("key", "value2")

        assert await storage1.get("key") == "value1"
        assert await storage2.get("key") == "value2"

    @pytest.mark.asyncio
    async def test_sqlite_storage_same_db_shares_data(self, tmp_path):
        """SQLiteStorage instances with same path should share data."""
        db_path = str(tmp_path / "shared.db")

        storage1 = SQLiteStorage(db_path)
        storage2 = SQLiteStorage(db_path)

        await storage1.set("key", "value")
        result = await storage2.get("key")
        assert result == "value"

        storage1.close()
        storage2.close()

    @pytest.mark.asyncio
    async def test_sqlite_storage_different_db_isolated(self, tmp_path):
        """SQLiteStorage instances with different paths should be isolated."""
        storage1 = SQLiteStorage(str(tmp_path / "db1.db"))
        storage2 = SQLiteStorage(str(tmp_path / "db2.db"))

        await storage1.set("key", "value1")
        await storage2.set("key", "value2")

        assert await storage1.get("key") == "value1"
        assert await storage2.get("key") == "value2"

        storage1.close()
        storage2.close()
