"""
Tests for semantic cache functionality.
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cyclic.agent import AgentResponse
from cyclic.cache import SemanticCache
from cyclic.sandbox import ExecutionResult


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_agent_response():
    """Standard AgentResponse for testing."""
    return AgentResponse(
        code="print('Hello, World!')",
        reasoning="Simple print statement",
        confidence=0.95,
    )


@pytest.fixture
def sample_execution_result():
    """Standard ExecutionResult for testing."""
    return ExecutionResult(
        stdout="Hello, World!\n",
        stderr="",
        exit_code=0,
    )


@pytest.fixture
def api_key():
    """Test API key."""
    return "test-api-key-12345"


class TestCoreCacheOperations:
    """Test core cache store and retrieve operations."""

    @pytest.mark.asyncio
    async def test_cache_stores_successful_execution(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that successful executions are stored in cache."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)
            await cache.store("print hello", sample_agent_response, sample_execution_result)

            stats = await cache.get_stats()
            assert stats["total_entries"] == 1

    @pytest.mark.asyncio
    async def test_cache_skips_failed_execution(
        self, temp_cache_dir, sample_agent_response, api_key
    ):
        """Test that failed executions are not cached."""
        failed_result = ExecutionResult(stdout="", stderr="Error", exit_code=1)

        cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)
        await cache.store("bad code", sample_agent_response, failed_result)

        stats = await cache.get_stats()
        assert stats["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_cache_hit_similar_prompt(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that similar prompts return cached results."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key, similarity_threshold=0.85)

            await cache.store("print hello world", sample_agent_response, sample_execution_result)

            cache_hit = await cache.search("print hello world")
            assert cache_hit is not None
            assert cache_hit.code == sample_agent_response.code
            assert 0.0 <= cache_hit.similarity <= 1.0
            assert cache_hit.similarity >= 0.85

    @pytest.mark.asyncio
    async def test_cache_miss_dissimilar_prompt(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that dissimilar prompts return None."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            store_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
            query_embedding = [0.0, 1.0, 0.0, 0.0, 0.0]

            def embedding_side_effect(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.data = [MagicMock()]
                if len(mock_embedding.call_args_list) == 1:
                    mock_response.data[0].embedding = store_embedding
                else:
                    mock_response.data[0].embedding = query_embedding
                return mock_response

            mock_embedding.side_effect = embedding_side_effect

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key, similarity_threshold=0.85)

            await cache.store("print hello", sample_agent_response, sample_execution_result)

            cache_hit = await cache.search("calculate factorial")
            assert cache_hit is None

    @pytest.mark.asyncio
    async def test_cache_returns_correct_cached_data(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that cached data matches what was stored."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(
                cache_dir=temp_cache_dir,
                api_key=api_key,
                store_prompt=True,
                store_outputs=True,
            )

            response = AgentResponse(
                code="x = 42\nprint(x)",
                reasoning="Set variable and print",
                confidence=0.88,
            )
            result = ExecutionResult(stdout="42\n", stderr="", exit_code=0)

            await cache.store("test prompt", response, result)
            cache_hit = await cache.search("test prompt")

            assert cache_hit is not None
            assert cache_hit.prompt == "test prompt"
            assert cache_hit.code == response.code
            assert cache_hit.reasoning == response.reasoning
            assert cache_hit.confidence == response.confidence
            assert cache_hit.execution_result.stdout == result.stdout
            assert cache_hit.execution_result.stderr == result.stderr
            assert cache_hit.execution_result.exit_code == result.exit_code


class TestDeterministicIdsAndUpsert:
    """Test deterministic IDs and upsert behavior."""

    @pytest.mark.asyncio
    async def test_deterministic_id_upsert(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that storing same (prompt, code) twice results in single entry with updated timestamp."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            prompt = "print hello"
            code1 = "print('hello')"

            response1 = AgentResponse(code=code1, reasoning="First", confidence=0.9)
            await cache.store(prompt, response1, sample_execution_result)

            stats1 = await cache.get_stats()
            assert stats1["total_entries"] == 1

            # Store same prompt+code again (should upsert, not create duplicate)
            response2 = AgentResponse(code=code1, reasoning="Second", confidence=0.95)
            await cache.store(prompt, response2, sample_execution_result)

            stats2 = await cache.get_stats()
            assert stats2["total_entries"] == 1

            # Verify the entry was updated (check reasoning changed)
            cache_hit = await cache.search(prompt)
            assert cache_hit is not None
            assert cache_hit.code == code1
            assert cache_hit.reasoning == "Second"
            assert cache_hit.confidence == 0.95

    @pytest.mark.asyncio
    async def test_different_code_creates_new_entry(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that different code with same prompt creates separate entries."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            prompt = "print hello"
            response1 = AgentResponse(code="print('hello')", reasoning="First", confidence=0.9)
            await cache.store(prompt, response1, sample_execution_result)

            response2 = AgentResponse(code="print('hi')", reasoning="Second", confidence=0.8)
            await cache.store(prompt, response2, sample_execution_result)

            stats = await cache.get_stats()
            assert stats["total_entries"] == 2

    @pytest.mark.asyncio
    async def test_doc_id_null_byte_robustness(
        self, temp_cache_dir, sample_execution_result, api_key
    ):
        """Prompts/code containing NUL must not collide under doc id hashing."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            r1 = AgentResponse(code="c", reasoning="r1", confidence=0.9)
            r2 = AgentResponse(code="b\0c", reasoning="r2", confidence=0.8)
            await cache.store("a\0b", r1, sample_execution_result)
            await cache.store("a", r2, sample_execution_result)

            stats = await cache.get_stats()
            assert stats["total_entries"] == 2


class TestCosineDistanceSpace:
    """Test cosine distance space pinning."""

    @pytest.mark.asyncio
    async def test_collection_has_cosine_distance_pinned(
        self, temp_cache_dir, api_key
    ):
        """Test that collection is created with cosine distance pinned."""
        cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)
        await cache._ensure_collection()

        metadata = getattr(cache.collection, "metadata", None) or {}
        assert metadata.get("hnsw:space") == "cosine"

    @pytest.mark.asyncio
    async def test_similarity_within_valid_range(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that computed similarity is always within [0, 1]."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key, similarity_threshold=0.0)

            await cache.store("test prompt", sample_agent_response, sample_execution_result)
            cache_hit = await cache.search("test prompt")

            assert cache_hit is not None
            assert 0.0 <= cache_hit.similarity <= 1.0

    @pytest.mark.asyncio
    async def test_similarity_scales_with_distance(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Orthogonal unit embeddings -> cosine distance 1 -> similarity 0.5."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            emb_store = [1.0, 0.0, 0.0]
            emb_query = [0.0, 1.0, 0.0]

            def embedding_side_effect(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.data = [MagicMock()]
                if len(mock_embedding.call_args_list) == 1:
                    mock_response.data[0].embedding = emb_store
                else:
                    mock_response.data[0].embedding = emb_query
                return mock_response

            mock_embedding.side_effect = embedding_side_effect

            cache = SemanticCache(
                cache_dir=temp_cache_dir, api_key=api_key, similarity_threshold=0.0
            )
            await cache.store("a", sample_agent_response, sample_execution_result)
            cache_hit = await cache.search("b")
            assert cache_hit is not None
            assert abs(cache_hit.similarity - 0.5) < 1e-5


class TestPrivacyDefaults:
    """Test privacy-safe persistence defaults."""

    @pytest.mark.asyncio
    async def test_default_does_not_store_prompt(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that prompt is not stored by default."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            prompt = "secret prompt with PII"
            await cache.store(prompt, sample_agent_response, sample_execution_result)

            cache_hit = await cache.search(prompt)
            assert cache_hit is not None
            assert cache_hit.prompt is None

    @pytest.mark.asyncio
    async def test_outputs_stored_by_default(
        self, temp_cache_dir, sample_agent_response, api_key
    ):
        """Test that stdout/stderr are stored by default."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            result = ExecutionResult(stdout="secret output", stderr="secret error", exit_code=0)
            await cache.store("test", sample_agent_response, result)

            cache_hit = await cache.search("test")
            assert cache_hit is not None
            assert cache_hit.execution_result.stdout == "secret output"
            assert cache_hit.execution_result.stderr == "secret error"

    @pytest.mark.asyncio
    async def test_opt_in_stores_prompt_and_outputs(
        self, temp_cache_dir, sample_agent_response, api_key
    ):
        """Test that opt-in flags enable storing prompt and outputs."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(
                cache_dir=temp_cache_dir,
                api_key=api_key,
                store_prompt=True,
                store_outputs=True,
            )

            prompt = "test prompt"
            result = ExecutionResult(stdout="output", stderr="error", exit_code=0)
            await cache.store(prompt, sample_agent_response, result)

            cache_hit = await cache.search(prompt)
            assert cache_hit is not None
            assert cache_hit.prompt == prompt
            assert cache_hit.execution_result.stdout == "output"
            assert cache_hit.execution_result.stderr == "error"


class TestEdgeCases:
    """Test edge cases and data integrity."""

    @pytest.mark.asyncio
    async def test_empty_cache_returns_none(self, temp_cache_dir, api_key):
        """Test that searching empty cache returns None."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3]
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)
            cache_hit = await cache.search("any prompt")
            assert cache_hit is None

    @pytest.mark.asyncio
    async def test_metadata_preservation(
        self, temp_cache_dir, api_key
    ):
        """Test that all metadata fields are preserved."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(
                cache_dir=temp_cache_dir,
                api_key=api_key,
                store_prompt=True,
                store_outputs=True,
            )

            response = AgentResponse(
                code="print('test')",
                reasoning="Test reasoning with special chars: !@#$%",
                confidence=0.75,
            )
            result = ExecutionResult(
                stdout="test\n",
                stderr="warning: test\n",
                exit_code=0,
            )

            await cache.store("test prompt", response, result)
            cache_hit = await cache.search("test prompt")

            assert cache_hit is not None
            assert cache_hit.reasoning == response.reasoning
            assert cache_hit.execution_result.stderr == result.stderr

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test handling of prompts with special characters."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(
                cache_dir=temp_cache_dir,
                api_key=api_key,
                store_prompt=True,
            )

            prompt = "print('Hello\nWorld\tTab')"
            await cache.store(prompt, sample_agent_response, sample_execution_result)
            cache_hit = await cache.search(prompt)

            assert cache_hit is not None
            assert cache_hit.prompt == prompt

    @pytest.mark.asyncio
    async def test_unicode_characters(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test handling of unicode characters."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(
                cache_dir=temp_cache_dir,
                api_key=api_key,
                store_prompt=True,
            )

            prompt = "打印('你好世界')"
            await cache.store(prompt, sample_agent_response, sample_execution_result)
            cache_hit = await cache.search(prompt)

            assert cache_hit is not None
            assert cache_hit.prompt == prompt


class TestErrorHandling:
    """Test error handling and resilience."""

    @pytest.mark.asyncio
    async def test_missing_api_key(self, temp_cache_dir):
        """Test that cache handles missing API key gracefully."""
        cache = SemanticCache(cache_dir=temp_cache_dir, api_key=None)

        cache_hit = await cache.search("test")
        assert cache_hit is None

        response = AgentResponse(code="print('test')", reasoning="test", confidence=0.9)
        result = ExecutionResult(stdout="test\n", stderr="", exit_code=0)
        await cache.store("test", response, result)

        stats = await cache.get_stats()
        assert stats["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_embedding_api_failure(self, temp_cache_dir, api_key):
        """Test that embedding API failures are handled gracefully."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            mock_embedding.side_effect = Exception("API Error")

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            cache_hit = await cache.search("test prompt")
            assert cache_hit is None

            response = AgentResponse(code="print('test')", reasoning="test", confidence=0.9)
            result = ExecutionResult(stdout="test\n", stderr="", exit_code=0)
            await cache.store("test", response, result)

            stats = await cache.get_stats()
            assert stats["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_empty_embedding_response(self, temp_cache_dir, api_key):
        """Test handling of empty embedding response."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            mock_response = MagicMock()
            mock_response.data = []
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            cache_hit = await cache.search("test")
            assert cache_hit is None

    @pytest.mark.asyncio
    async def test_get_embedding_does_not_retry_auth_error(self, temp_cache_dir, api_key):
        from litellm.exceptions import AuthenticationError

        with patch("cyclic.cache.aembedding", new_callable=AsyncMock) as mock_emb:
            mock_emb.side_effect = AuthenticationError(
                "bad key", llm_provider="openai", model="text-embedding-3-small"
            )
            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)
            await cache._ensure_collection()

            with pytest.raises(AuthenticationError):
                await cache._get_embedding("x")

            assert mock_emb.call_count == 1


class TestCacheManagement:
    """Test cache management operations."""

    @pytest.mark.asyncio
    async def test_clear_cache(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that clear removes all entries."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            await cache.store("test", sample_agent_response, sample_execution_result)
            stats1 = await cache.get_stats()
            assert stats1["total_entries"] == 1

            await cache.clear()
            stats2 = await cache.get_stats()
            assert stats2["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_cache_stats_persistent_only(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that cache stats only report persistent stats."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            await cache.store("prompt", sample_agent_response, sample_execution_result)

            stats = await cache.get_stats()
            assert "total_entries" in stats
            assert "cache_dir" in stats
            assert "cache_size_bytes" in stats
            assert stats["total_entries"] == 1
            assert "hits" not in stats
            assert "misses" not in stats
            assert "hit_rate" not in stats

    @pytest.mark.asyncio
    async def test_cache_persistence(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that cache persists across instances."""
        with patch("cyclic.cache.aembedding") as mock_embedding:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            cache1 = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)
            await cache1.store("persistent test", sample_agent_response, sample_execution_result)
            del cache1

            cache2 = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)
            cache_hit = await cache2.search("persistent test")

            assert cache_hit is not None
            assert cache_hit.code == sample_agent_response.code


class TestAsyncSafety:
    """Test async/threading safety."""

    @pytest.mark.asyncio
    async def test_chroma_operations_use_thread_pool(
        self, temp_cache_dir, sample_agent_response, sample_execution_result, api_key
    ):
        """Test that ChromaDB operations are executed in thread pool."""
        with patch("cyclic.cache.aembedding") as mock_embedding, \
             patch("asyncio.to_thread") as mock_to_thread:
            embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = embedding
            mock_embedding.return_value = mock_response

            # Make to_thread actually call the function (passthrough)
            async def to_thread_passthrough(func, *args, **kwargs):
                return func(*args, **kwargs)

            mock_to_thread.side_effect = to_thread_passthrough

            cache = SemanticCache(cache_dir=temp_cache_dir, api_key=api_key)

            await cache.store("test", sample_agent_response, sample_execution_result)

            # Verify to_thread was called for Chroma operations
            assert mock_to_thread.called

            stats = await cache.get_stats()
            assert stats["total_entries"] == 1
