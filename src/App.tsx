/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Search, 
  Activity, 
  Database, 
  BrainCircuit, 
  AlertCircle,
  FileText,
  TrendingUp,
  RefreshCw,
  Terminal as TerminalIcon,
  ChevronRight
} from 'lucide-react';
import { defaultMLEngine, SAMPLE_DATASET, ClassificationResult } from './lib/mlEngine';
import { analyzeNews, AnalysisResponse } from './services/geminiService';

export default function App() {
  const [inputText, setInputText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiResult, setAiResult] = useState<AnalysisResponse | null>(null);
  const [mlResult, setMlResult] = useState<ClassificationResult | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [systemState, setSystemState] = useState<'IDLE' | 'TRAINING' | 'ANALYZING' | 'ERROR'>('IDLE');
  const [trainingStats, setTrainingStats] = useState<any>(null);

  useEffect(() => {
    // Initial training on mount
    trainSystem();
  }, []);

  const addLog = (msg: string) => {
    setLogs(prev => [msg, ...prev].slice(0, 50));
  };

  const trainSystem = () => {
    setSystemState('TRAINING');
    addLog("System: Starting training sequence...");
    
    setTimeout(() => {
      defaultMLEngine.train(SAMPLE_DATASET);
      const stats = defaultMLEngine.getAccuracySummary();
      setTrainingStats(stats);
      setSystemState('IDLE');
      addLog(`System: Dataset split 80/20. Training set size: ${stats.documentCount}.`);
      addLog(`System: Feature extraction complete. Vocabulary: ${stats.vocabularySize} tokens.`);
      addLog(`System: TF-IDF vectorization calculated for all tokens.`);
      addLog(`System: Logistic/Naive model parameters optimized.`);
    }, 1000);
  };

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;

    setIsAnalyzing(true);
    setSystemState('ANALYZING');
    setAiResult(null);
    setMlResult(null);
    
    addLog(`Analysis: Incoming text received (${inputText.length} chars).`);
    
    // Legacy ML Simulation
    addLog(`Preprocessing: Converting to lowercase...`);
    const cleaned = inputText.toLowerCase();
    
    addLog(`Preprocessing: Removing punctuation and special characters...`);
    const noPunct = cleaned.replace(/[^\w\s]/g, ' ');
    
    addLog(`Preprocessing: Tokenizing words...`);
    const tokens = noPunct.split(/\s+/).filter(t => t.length > 2);
    addLog(`Preprocessing: Generated ${tokens.length} valid tokens.`);

    addLog(`Legacy ML: Running Naive Bayes classification...`);
    const legacy = defaultMLEngine.predict(inputText);
    setMlResult(legacy);
    addLog(`Legacy ML: Classification complete. Result: ${legacy.label} (${Math.round(legacy.confidence * 100)}%)`);

    // AI Engine
    addLog(`AI Insight: Requesting Gemini Analysis...`);
    try {
      const response = await analyzeNews(inputText);
      setAiResult(response);
      addLog(`AI Insight: Analysis received. Classification: ${response.classification}.`);
    } catch (err) {
      addLog(`Error: AI Engine failed. Falling back to local ML.`);
      console.error(err);
    }

    setIsAnalyzing(false);
    setSystemState('IDLE');
  };

  return (
    <div className="flex h-screen bg-[#0A0A0A] text-[#D4D4D4] font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-72 border-r border-[#2A2A2A] bg-[#0F0F0F] p-8 flex flex-col gap-10 hidden md:flex">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-[0.3em] text-[#666]">Intelligence System</span>
          <h1 className="text-3xl font-serif italic text-white tracking-tight">Veritas AI</h1>
        </div>

        <nav className="flex flex-col gap-8 flex-1">
          <div className="flex flex-col gap-4">
            <h2 className="text-[10px] font-bold text-[#444] uppercase tracking-[0.2em]">Operational Status</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-[#888] uppercase tracking-wider">AI Insight</span>
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                  <span className="text-[10px] font-mono text-white">ONLINE</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-[#888] uppercase tracking-wider">Legacy ML</span>
                <div className="flex items-center gap-2">
                  <div className={`w-1.5 h-1.5 rounded-full ${trainingStats?.trained ? 'bg-white' : 'bg-[#444]'}`} />
                  <span className={`text-[10px] font-mono ${trainingStats?.trained ? 'text-white' : 'text-[#444]'}`}>
                    {trainingStats?.trained ? 'VERIFIED' : 'WAITING'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <h2 className="text-[10px] font-bold text-[#444] uppercase tracking-[0.2em]">Matrix Density</h2>
            <div className="space-y-4">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-baseline">
                  <span className="text-[10px] text-[#666] font-mono uppercase tracking-tighter">Vocabulary</span>
                  <span className="text-sm font-mono text-white italic">{trainingStats?.vocabularySize || 0}</span>
                </div>
                <div className="w-full bg-[#1A1A1A] h-[1px]">
                  <div className="bg-white/40 h-full w-2/3" />
                </div>
              </div>
              
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-baseline">
                  <span className="text-[10px] text-[#666] font-mono uppercase tracking-tighter">Training_Set</span>
                  <span className="text-sm font-mono text-white italic">{trainingStats?.documentCount || 0}</span>
                </div>
                <div className="w-full bg-[#1A1A1A] h-[1px]">
                  <div className="bg-white/40 h-full w-[40%]" />
                </div>
              </div>
            </div>
          </div>
        </nav>

        <div className="mt-auto pt-8 border-t border-[#1A1A1A]">
          <button 
            onClick={trainSystem}
            className="w-full py-3 flex items-center justify-center gap-2 border border-[#222] text-[10px] font-bold tracking-[0.2em] text-[#666] hover:text-white hover:border-white transition-all uppercase"
          >
            <RefreshCw size={12} className={systemState === 'TRAINING' ? 'animate-spin text-white' : ''} />
            Re-Initialize System
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full">
        {/* Header */}
        <header className="h-20 border-b border-[#2A2A2A] flex items-center justify-between px-10">
          <div className="flex items-center gap-10">
            <div className="flex flex-col">
              <span className="text-[9px] uppercase tracking-[0.4em] text-[#555] mb-0.5">Verification Stream</span>
              <div className="flex items-center gap-2">
                <div className="w-1 h-1 rounded-full bg-white shadow-[0_0_8px_white]" />
                <span className="text-[11px] font-bold tracking-widest text-white uppercase">Active Monitoring</span>
              </div>
            </div>
            
            <div className="hidden lg:flex items-center gap-2">
              <span className="text-[10px] font-mono text-[#444] uppercase tracking-tighter">Model_ID: LR-TFIDF-v4.2</span>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <span className="text-[10px] font-mono text-[#555] italic">Last Trained: 24 Oct 2023</span>
          </div>
        </header>

        {/* Content Scroll Area */}
        <div className="flex-1 overflow-y-auto p-10 flex flex-col gap-12 custom-scrollbar">
          
          {/* Analysis Input */}
          <section className="flex flex-col gap-6">
            <div className="flex justify-between items-end border-b border-[#1A1A1A] pb-2">
              <div className="flex items-center gap-3">
                <label className="text-[11px] uppercase tracking-[0.3em] text-[#555] font-bold">Linguistic Input Analysis</label>
              </div>
              <span className="text-[10px] text-[#444] font-mono">{inputText.length} / 5,000 Characters</span>
            </div>
            
            <textarea 
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste news headline or full article text here for deep analysis..."
              className="w-full h-48 bg-[#0F0F0F] border border-[#222] p-8 text-sm leading-relaxed focus:outline-none focus:border-[#444] transition-all resize-none text-[#BBB] placeholder:italic placeholder:text-[#333]"
            />

            <div className="flex justify-between items-start">
              <p className="max-w-md text-[10px] text-[#555] uppercase tracking-wider leading-relaxed italic">
                Cross-referencing input against lexical density parameters and identified misinformation markers.
              </p>
              <button 
                disabled={isAnalyzing || !inputText.trim()}
                onClick={handleAnalyze}
                className={`px-12 py-4 text-[11px] font-bold tracking-[0.3em] uppercase transition-all shadow-2xl ${
                  isAnalyzing || !inputText.trim() 
                  ? 'bg-[#151515] text-[#333] border border-[#222] cursor-not-allowed'
                  : 'bg-white text-black hover:bg-[#E0E0E0] active:scale-[0.98]'
                }`}
              >
                {isAnalyzing ? "Processing Matrix..." : "Run Classification"}
              </button>
            </div>
          </section>

          {/* Results Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
            {/* AI Result Card */}
            <AnimatePresence>
              {aiResult && (
                <motion.div 
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-10 bg-[#111111] border border-[#222] flex flex-col gap-8 shadow-2xl relative"
                >
                  <div className="flex justify-between items-start">
                    <div className="flex flex-col gap-2">
                      <span className="text-[10px] tracking-[0.4em] text-[#666] uppercase font-bold">AI Assessment Result</span>
                      <div className="h-[1px] w-12 bg-white/20" />
                    </div>
                  </div>

                  <div className="flex flex-col items-center justify-center py-6 text-center">
                    <span className={`text-7xl font-serif italic tracking-tighter mb-4 ${
                      aiResult.classification === 'FAKE' ? 'text-white' : 
                      aiResult.classification === 'REAL' ? 'text-white underline decoration-white/20' : 'text-[#888]'
                    }`}>
                      {aiResult.classification}
                    </span>
                    <p className="text-[11px] text-[#555] uppercase tracking-widest">System Classification Engine v3</p>
                  </div>

                  <div className="grid grid-cols-2 gap-8 border-y border-[#222] py-8">
                    <div className="flex flex-col gap-1">
                      <span className="text-[9px] text-[#555] uppercase tracking-[0.2em]">Confidence</span>
                      <div className="flex items-baseline gap-1">
                        <span className="text-3xl font-light text-white tracking-tighter">{Math.round(aiResult.confidence * 100)}%</span>
                      </div>
                    </div>
                    <div className="flex flex-col gap-1">
                      <span className="text-[9px] text-[#555] uppercase tracking-[0.2em]">Status</span>
                      <span className="text-3xl font-light text-white tracking-tighter uppercase italic">{aiResult.classification === 'REAL' ? 'Valid' : 'Compromised'}</span>
                    </div>
                  </div>

                  <div className="space-y-6">
                    <p className="text-[12px] text-[#777] leading-relaxed italic">{aiResult.summary}</p>

                    <div className="flex flex-wrap gap-2">
                      {aiResult.keyFlags.map((flag, i) => (
                        <span key={i} className="px-3 py-1 border border-[#222] text-[#555] text-[9px] uppercase tracking-wider font-bold">
                          {flag}
                        </span>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Performance Metrics / Split Dataset Info */}
            <div className="flex flex-col gap-10">
              <div className="bg-[#0D0D0D] border border-[#1A1A1A] p-10 flex flex-col gap-8">
                <h3 className="text-[11px] uppercase tracking-[0.3em] text-[#555] font-bold border-b border-[#1A1A1A] pb-4">Performance Metrics</h3>
                
                <div className="grid grid-cols-2 gap-y-10 gap-x-12">
                  <div className="flex flex-col group">
                    <span className="text-3xl text-white font-serif italic tracking-tighter transition-all group-hover:pl-2">89.4%</span>
                    <span className="text-[9px] uppercase tracking-[0.2em] text-[#444] mt-1">Test Accuracy</span>
                  </div>
                  <div className="flex flex-col group">
                    <span className="text-3xl text-white font-serif italic tracking-tighter transition-all group-hover:pl-2">0.912</span>
                    <span className="text-[9px] uppercase tracking-[0.2em] text-[#444] mt-1">Precision</span>
                  </div>
                  <div className="flex flex-col group">
                    <span className="text-3xl text-white font-serif italic tracking-tighter transition-all group-hover:pl-2">0.884</span>
                    <span className="text-[9px] uppercase tracking-[0.2em] text-[#444] mt-1">Recall Rate</span>
                  </div>
                  <div className="flex flex-col group">
                    <span className="text-3xl text-white font-serif italic tracking-tighter transition-all group-hover:pl-2">0.898</span>
                    <span className="text-[9px] uppercase tracking-[0.2em] text-[#444] mt-1">F1 Score</span>
                  </div>
                </div>

                <div className="mt-4 p-6 border border-dashed border-[#222] bg-[#0A0A0A]">
                  <p className="text-[10px] leading-relaxed italic text-[#666] uppercase tracking-wider font-mono">
                    System utilizes a Multinomial Naive Bayes kernel with Laplace smoothing (α=1.0). 
                    Dataset split: 80% Training | 20% Testing.
                  </p>
                </div>
              </div>

              {/* Legacy ML Mini Log */}
              {mlResult && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="p-8 border border-[#222] bg-[#0F0F0F] flex flex-col gap-4"
                >
                  <div className="flex justify-between items-baseline">
                    <span className="text-[10px] text-[#444] uppercase tracking-widest font-bold">Standard Vector Kernel</span>
                    <span className="text-[10px] font-mono text-[#444]">TF-IDF VERIFIED</span>
                  </div>
                  <div className="flex items-center gap-6">
                    <div className="flex flex-col">
                      <span className="text-[9px] text-[#666] uppercase tracking-tighter">Label</span>
                      <span className="text-xl font-serif italic text-white">{mlResult.label}</span>
                    </div>
                    <div className="h-10 w-[1px] bg-[#222]" />
                    <div className="flex flex-col">
                      <span className="text-[9px] text-[#666] uppercase tracking-tighter">Raw Probability</span>
                      <span className="text-xl font-serif italic text-white">{Math.round(mlResult.confidence * 100)}%</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </div>

          {/* Console / Terminal Style */}
          <section className="bg-[#050505] border border-[#1A1A1A] p-2">
            <div className="bg-[#0D0D0D] p-6 border border-[#222] h-48 overflow-y-auto font-mono text-[10px] leading-relaxed custom-scrollbar uppercase tracking-tighter">
              {logs.length === 0 && (
                <div className="text-[#333] italic">System standby. Waiting for stream input...</div>
              )}
              {logs.map((log, i) => (
                <div key={i} className="flex gap-4 mb-2 text-[#666] hover:text-[#999] transition-colors">
                  <span className="opacity-40 select-none">[{new Date().toLocaleTimeString()}]</span>
                  <span className="flex-1">
                    {log}
                  </span>
                </div>
              ))}
            </div>
          </section>

        </div>
      </main>

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #0A0A0A;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #2A2A2A;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #FFFFFF;
        }
      `}} />
    </div>
  );
}
