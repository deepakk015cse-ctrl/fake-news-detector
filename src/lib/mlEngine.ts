/**
 * Simple Machine Learning engine for Text Classification (Naive Bayes + TF-IDF)
 * Translated for browser environments.
 */

export interface ClassificationResult {
  label: 'REAL' | 'FAKE';
  confidence: number;
}

export class MLEngine {
  private vocabulary: Set<string> = new Set();
  private documents: { text: string; label: string }[] = [];
  private trained: boolean = false;

  // Model parameters
  private classProbabilities: Record<string, number> = {};
  private wordProbabilities: Record<string, Record<string, number>> = {};
  private labels: string[] = ['REAL', 'FAKE'];

  /**
   * Preprocess text: Lowercase, remove punctuation, tokenize
   */
  preprocess(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2); // Filter small words (simple stopword removal)
  }

  /**
   * Train the model with a dataset
   */
  train(data: { text: string; label: 'REAL' | 'FAKE' }[]) {
    // Split dataset (80% training, 20% testing)
    const splitIndex = Math.floor(data.length * 0.8);
    const trainingData = data.slice(0, splitIndex);
    const testingData = data.slice(splitIndex);

    this.documents = trainingData;
    const documentCount = trainingData.length;
    
    // Reset model
    this.vocabulary = new Set();
    const labelCounts: Record<string, number> = { REAL: 0, FAKE: 0 };
    const wordCounts: Record<string, Record<string, number>> = { REAL: {}, FAKE: {} };
    const totalWordsByLabel: Record<string, number> = { REAL: 0, FAKE: 0 };
    const docFrequency: Record<string, number> = {};

    // First pass: Build vocabulary and document frequency for TF-IDF
    trainingData.forEach(doc => {
      const tokens = Array.from(new Set(this.preprocess(doc.text))); // Unique words in doc
      tokens.forEach(word => {
        docFrequency[word] = (docFrequency[word] || 0) + 1;
        this.vocabulary.add(word);
      });
    });

    // Second pass: Count frequencies
    trainingData.forEach(doc => {
      labelCounts[doc.label]++;
      const tokens = this.preprocess(doc.text);
      tokens.forEach(word => {
        // Apply TF-IDF logic: (Frequency) * log(TotalDocs / DocFrequency)
        // For Naive Bayes, we often use weighted frequencies or just counts.
        // We'll use weighted counts to simulate TF-IDF influence.
        const idf = Math.log(documentCount / (docFrequency[word] || 1));
        const weight = 1 + idf; // TF-IDF representation
        
        wordCounts[doc.label][word] = (wordCounts[doc.label][word] || 0) + weight;
        totalWordsByLabel[doc.label] += weight;
      });
    });

    // Calculate probabilities with Laplace smoothing
    this.labels.forEach(label => {
      this.classProbabilities[label] = labelCounts[label] / documentCount;
      this.wordProbabilities[label] = {};
      
      const vocabularySize = this.vocabulary.size;
      this.vocabulary.forEach(word => {
        const count = wordCounts[label][word] || 0;
        // Laplace smoothing: (count + 1) / (total words in label + vocabulary size)
        this.wordProbabilities[label][word] = (count + 1) / (totalWordsByLabel[label] + vocabularySize);
      });
    });

    this.trained = true;
  }

  /**
   * Predict the label of a text
   */
  predict(text: string): ClassificationResult {
    if (!this.trained) {
      // Return a default if not trained
      return { label: 'REAL', confidence: 0.5 };
    }

    const tokens = this.preprocess(text);
    const scores: Record<string, number> = {};

    this.labels.forEach(label => {
      // Use log probabilities to avoid underflow
      let logScore = Math.log(this.classProbabilities[label]);
      
      tokens.forEach(word => {
        if (this.vocabulary.has(word)) {
          logScore += Math.log(this.wordProbabilities[label][word]);
        } else {
          // Probability for unknown words (Laplace smoothing)
          const vocabularySize = this.vocabulary.size;
          logScore += Math.log(1 / (vocabularySize + 1));
        }
      });
      
      scores[label] = logScore;
    });

    // Convert log scores back to probabilities (softmax-like)
    const maxLog = Math.max(...Object.values(scores));
    const expScores = Object.fromEntries(
      Object.entries(scores).map(([label, score]) => [label, Math.exp(score - maxLog)])
    );
    
    const sumExp = Object.values(expScores).reduce((a, b) => a + b, 0);
    const probabilities = Object.fromEntries(
      Object.entries(expScores).map(([label, exp]) => [label, exp / sumExp])
    );

    const winner = probabilities['REAL'] > probabilities['FAKE'] ? 'REAL' : 'FAKE';
    
    return {
      label: winner as 'REAL' | 'FAKE',
      confidence: probabilities[winner]
    };
  }

  getAccuracySummary() {
    return {
      vocabularySize: this.vocabulary.size,
      documentCount: this.documents.length,
      trained: this.trained
    };
  }
}

export const defaultMLEngine = new MLEngine();

// Sample dataset for priming the detector
export const SAMPLE_DATASET: { text: string; label: 'REAL' | 'FAKE' }[] = [
  { text: "Scientific study shows that drinking water is healthy for humans.", label: "REAL" },
  { text: "NASA confirms new planet discovered in Andromeda galaxy.", label: "REAL" },
  { text: "Local government implements new park safety guidelines.", label: "REAL" },
  { text: "The stock market went up by 2% today after economic report.", label: "REAL" },
  { text: "Scientists announce breakthrough in renewable energy storage.", label: "REAL" },
  { text: "World leaders meet to discuss climate change initiatives.", label: "REAL" },
  { text: "SpaceX successfully launches another set of Starlink satellites.", label: "REAL" },
  { text: "Health experts recommend 8 hours of sleep for optimal function.", label: "REAL" },
  { text: "Technology giant unveils new smartphone with revolutionary camera.", label: "REAL" },
  { text: "Global health organization declares end of seasonal flu alert.", label: "REAL" },
  
  { text: "Alien lizards have infiltrated the Senate and are eating all the chips.", label: "FAKE" },
  { text: "Secret study reveals that the moon is actually made of blue cheese.", label: "FAKE" },
  { text: "Eating spoons every morning will make you live for 500 years.", label: "FAKE" },
  { text: "Invisible rays from your microwave are stealing your memories.", label: "FAKE" },
  { text: "Giant flying squirrels have taken over the city hall of Paris.", label: "FAKE" },
  { text: "Drinking gold liquid will give you the ability to fly through walls.", label: "FAKE" },
  { text: "The government is replacing all birds with robotic surveillance drones.", label: "FAKE" },
  { text: "New law makes it illegal to breathe on Tuesdays without a permit.", label: "FAKE" },
  { text: "Magic rocks found in desert can solve all your financial problems.", label: "FAKE" },
  { text: "Clouds are actually giant cotton candy machines operated by sky people.", label: "FAKE" },
];
