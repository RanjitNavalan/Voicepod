import { useState, useEffect, useRef } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { Upload, Mic, Download, Sparkles, Music, Wand2, Radio } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Home = () => {
  const [selectedPreset, setSelectedPreset] = useState("podcast_calm");
  const [presets, setPresets] = useState({});
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState("");
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [showTranscript, setShowTranscript] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  // Fetch presets
  useEffect(() => {
    const fetchPresets = async () => {
      try {
        const response = await axios.get(`${API}/presets`);
        setPresets(response.data);
      } catch (error) {
        console.error("Error fetching presets:", error);
        toast.error("Failed to load presets");
      }
    };
    fetchPresets();
  }, []);

  // Poll for job status
  useEffect(() => {
    if (!jobId || !processing) return;

    const pollStatus = async () => {
      try {
        const response = await axios.get(`${API}/status/${jobId}`);
        const { status, progress: prog, current_step, download_url, error } = response.data;

        setProgress(prog);
        setCurrentStep(current_step);

        if (status === "completed") {
          setProcessing(false);
          setDownloadUrl(download_url);
          toast.success("Voicepod is ready! 🎉");
        } else if (status === "failed") {
          setProcessing(false);
          toast.error(`Processing failed: ${error}`);
        }
      } catch (error) {
        console.error("Error polling status:", error);
      }
    };

    const interval = setInterval(pollStatus, 2000);
    return () => clearInterval(interval);
  }, [jobId, processing]);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      setAudioBlob(null);
      toast.success(`Selected: ${file.name}`);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      setUploadedFile(file);
      setAudioBlob(null);
      toast.success(`Selected: ${file.name}`);
    } else {
      toast.error("Please drop an audio file");
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/mp3' });
        setAudioBlob(blob);
        setUploadedFile(null);
        toast.success("Recording saved!");
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      toast.info("Recording started...");
    } catch (error) {
      toast.error("Microphone access denied");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const handleProcess = async () => {
    const fileToUpload = uploadedFile || (audioBlob ? new File([audioBlob], "recording.mp3") : null);
    
    if (!fileToUpload) {
      toast.error("Please upload or record audio first");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileToUpload);
    formData.append("preset", selectedPreset);

    try {
      setProcessing(true);
      setProgress(0);
      setDownloadUrl(null);
      
      const response = await axios.post(`${API}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setJobId(response.data.job_id);
      toast.success("Processing started!");
    } catch (error) {
      setProcessing(false);
      toast.error("Upload failed: " + error.message);
    }
  };

  const handleDownload = () => {
    if (downloadUrl) {
      window.location.href = `${BACKEND_URL}${downloadUrl}`;
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <div className="text-center space-y-6">
            <h1 className="hero-title" data-testid="hero-title">
              Voicepod Creator Studio
            </h1>
            <p className="hero-subtitle" data-testid="hero-subtitle">
              Transform raw audio into broadcast-quality Voicepods in under 60 seconds
            </p>
            <div className="flex items-center justify-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-cyan-400" />
                <span>AI-Powered Cleanup</span>
              </div>
              <div className="flex items-center gap-2">
                <Music className="w-5 h-5 text-purple-400" />
                <span>Auto Music Mix</span>
              </div>
              <div className="flex items-center gap-2">
                <Wand2 className="w-5 h-5 text-pink-400" />
                <span>Emotion Detection</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-5xl mx-auto space-y-8">
          
          {/* Preset Selection */}
          <Card className="card-glass" data-testid="preset-selector">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Radio className="w-5 h-5" />
                Choose Your Style
              </CardTitle>
              <CardDescription>Select a processing preset for your audio</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={selectedPreset} onValueChange={setSelectedPreset}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="podcast_calm" data-testid="preset-podcast-calm">
                    Podcast Calm
                  </TabsTrigger>
                  <TabsTrigger value="dramatic" data-testid="preset-dramatic">
                    Dramatic
                  </TabsTrigger>
                  <TabsTrigger value="ai_narrator" data-testid="preset-ai-narrator">
                    AI Narrator
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="podcast_calm" className="mt-4">
                  <div className="preset-card">
                    <h4 className="font-semibold mb-2">Professional Podcast Quality</h4>
                    <p className="text-sm text-gray-400">
                      Clean audio with subtle ambient music. Perfect for conversational content.
                    </p>
                  </div>
                </TabsContent>
                <TabsContent value="dramatic" className="mt-4">
                  <div className="preset-card">
                    <h4 className="font-semibold mb-2">Cinematic & Impactful</h4>
                    <p className="text-sm text-gray-400">
                      Enhanced studio sound with dramatic music accents. Ideal for storytelling.
                    </p>
                  </div>
                </TabsContent>
                <TabsContent value="ai_narrator" className="mt-4">
                  <div className="preset-card">
                    <h4 className="font-semibold mb-2">AI-Powered Studio Voice</h4>
                    <p className="text-sm text-gray-400">
                      Professional AI narrator voice with ambient music. Broadcast-ready quality.
                    </p>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Upload Section */}
          <Card className="card-glass" data-testid="upload-section">
            <CardHeader>
              <CardTitle>Upload or Record Audio</CardTitle>
              <CardDescription>30-90 second audio clips work best</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Drop Zone */}
              <div
                className="upload-zone"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
                data-testid="upload-dropzone"
              >
                <Upload className="w-12 h-12 mb-4" />
                <p className="text-lg font-medium mb-2">
                  {uploadedFile ? uploadedFile.name : audioBlob ? "Recording ready" : "Drop audio file or click to browse"}
                </p>
                <p className="text-sm text-gray-400">Supports MP3, WAV, M4A, OGG, MPEG, WebM</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  data-testid="file-input"
                />
              </div>

              {/* Record Button */}
              <div className="flex justify-center">
                <Button
                  variant={isRecording ? "destructive" : "outline"}
                  size="lg"
                  onClick={isRecording ? stopRecording : startRecording}
                  className="gap-2"
                  data-testid="record-button"
                >
                  <Mic className="w-5 h-5" />
                  {isRecording ? "Stop Recording" : "Record Audio"}
                </Button>
              </div>

              {/* Process Button */}
              <Button
                size="lg"
                className="w-full process-button"
                onClick={handleProcess}
                disabled={processing || (!uploadedFile && !audioBlob)}
                data-testid="process-button"
              >
                <Sparkles className="w-5 h-5 mr-2" />
                {processing ? "Processing..." : "Create Voicepod"}
              </Button>
            </CardContent>
          </Card>

          {/* Processing Status */}
          {processing && (
            <Card className="card-glass" data-testid="processing-status">
              <CardHeader>
                <CardTitle>Processing Your Voicepod</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Progress value={progress} className="h-3" data-testid="progress-bar" />
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400" data-testid="current-step">{currentStep}</span>
                  <span className="font-semibold" data-testid="progress-percentage">{progress}%</span>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Download Section */}
          {downloadUrl && (
            <Card className="card-glass download-card" data-testid="download-section">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-yellow-400" />
                  Your Voicepod is Ready!
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Button
                  size="lg"
                  className="w-full download-button"
                  onClick={handleDownload}
                  data-testid="download-button"
                >
                  <Download className="w-5 h-5 mr-2" />
                  Download Voicepod
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;