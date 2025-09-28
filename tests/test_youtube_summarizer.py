import unittest
from unittest import mock
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

import YouTube_Summarizer as ys


VALID_URL = "https://www.youtube.com/watch?v=abcdefghijk"


def make_transcript(
    language_code,
    *,
    is_generated=False,
    fetch_result=None,
    translation_languages=None,
    translated_fetch_result=None,
):
    transcript = mock.Mock()
    transcript.language_code = language_code
    transcript.is_generated = is_generated
    transcript.translation_languages = translation_languages or []
    transcript.fetch.return_value = fetch_result if fetch_result is not None else []
    translated = mock.Mock()
    translated.fetch.return_value = (
        translated_fetch_result if translated_fetch_result is not None else []
    )
    transcript.translate.return_value = translated
    return transcript


class GetVideoIdTests(unittest.TestCase):
    def test_extracts_from_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=abc123DEF45&feature=share"
        self.assertEqual(ys.get_video_id(url), "abc123DEF45")

    def test_extracts_from_short_url(self):
        url = "https://youtu.be/abc123DEF45"
        self.assertEqual(ys.get_video_id(url), "abc123DEF45")

    def test_extracts_from_shorts_url(self):
        url = "https://www.youtube.com/shorts/abc123DEF45"
        self.assertEqual(ys.get_video_id(url), "abc123DEF45")

    def test_returns_none_for_invalid_url(self):
        self.assertIsNone(ys.get_video_id("https://www.example.com/watch"))


class GetTranscriptTests(unittest.TestCase):
    @mock.patch("YouTube_Summarizer.YouTubeTranscriptApi")
    def test_returns_manual_transcript_when_available(self, mock_api):
        manual_payload = [{"text": "manual"}]
        auto_transcript = make_transcript("en", is_generated=True, fetch_result=[{"text": "auto"}])
        manual_transcript = make_transcript("en", fetch_result=manual_payload)
        mock_api.return_value.list.return_value = [auto_transcript, manual_transcript]

        transcript = ys.get_transcript(VALID_URL)

        self.assertEqual(transcript, manual_payload)

    @mock.patch("YouTube_Summarizer.YouTubeTranscriptApi")
    def test_returns_auto_transcript_when_manual_missing(self, mock_api):
        auto_payload = [{"text": "auto"}]
        auto_transcript = make_transcript(
            "en-US", is_generated=True, fetch_result=auto_payload
        )
        mock_api.return_value.list.return_value = [auto_transcript]

        transcript = ys.get_transcript(VALID_URL)

        self.assertEqual(transcript, auto_payload)

    @mock.patch("YouTube_Summarizer.YouTubeTranscriptApi")
    def test_translates_non_english_transcript_when_supported(self, mock_api):
        translated_payload = [{"text": "translated"}]
        foreign_transcript = make_transcript(
            "es",
            translation_languages=[{"language_code": "en"}],
            translated_fetch_result=translated_payload,
        )
        mock_api.return_value.list.return_value = [foreign_transcript]

        transcript = ys.get_transcript(VALID_URL)

        self.assertEqual(transcript, translated_payload)
        foreign_transcript.translate.assert_called_once_with("en")

    @mock.patch("YouTube_Summarizer.YouTubeTranscriptApi")
    def test_prefers_specific_english_variant_when_only_variant_available(self, mock_api):
        translated_payload = [{"text": "translated"}]
        foreign_transcript = make_transcript(
            "fr",
            translation_languages=[{"language_code": "en-US"}],
            translated_fetch_result=translated_payload,
        )
        mock_api.return_value.list.return_value = [foreign_transcript]

        transcript = ys.get_transcript(VALID_URL)

        self.assertEqual(transcript, translated_payload)
        foreign_transcript.translate.assert_called_once_with("en-US")

    @mock.patch("YouTube_Summarizer.YouTubeTranscriptApi")
    def test_returns_none_when_service_raises_known_exception(self, mock_api):
        mock_api.return_value.list.side_effect = NoTranscriptFound(
            video_id="abcdefghijk",
            requested_language_codes=["en"],
            transcript_data={}
        )

        transcript = ys.get_transcript(VALID_URL)
        self.assertIsNone(transcript)

    @mock.patch("YouTube_Summarizer.YouTubeTranscriptApi")
    def test_returns_none_when_transcripts_disabled(self, mock_api):
        mock_api.return_value.list.side_effect = TranscriptsDisabled("Transcripts disabled")
        self.assertIsNone(ys.get_transcript(VALID_URL))

    @mock.patch("YouTube_Summarizer.YouTubeTranscriptApi")
    def test_returns_none_when_video_unavailable(self, mock_api):
        mock_api.return_value.list.side_effect = VideoUnavailable("Video unavailable")
        self.assertIsNone(ys.get_transcript(VALID_URL))


class PreferredTranslationCodeTests(unittest.TestCase):
    def test_returns_none_when_no_languages(self):
        transcript = mock.Mock(translation_languages=None)
        self.assertIsNone(ys._preferred_translation_code(transcript))

    def test_prefers_plain_english_code(self):
        transcript = mock.Mock(
            translation_languages=[
                {"language_code": "es"},
                {"language_code": "en"},
                {"language_code": "en-US"},
            ]
        )
        self.assertEqual(ys._preferred_translation_code(transcript), "en")

    def test_returns_first_variant_when_plain_not_available(self):
        transcript = mock.Mock(
            translation_languages=[
                {"language_code": "es"},
                {"language_code": "en-GB"},
                {"language_code": "en-US"},
            ]
        )
        self.assertEqual(ys._preferred_translation_code(transcript), "en-GB")


if __name__ == "__main__":
    unittest.main()
