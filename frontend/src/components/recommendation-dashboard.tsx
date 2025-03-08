import React, { useState } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

// Sample data for demonstration purposes
const modelPerformance = [
  { name: 'Popularity (count)', precision: 0.024, recall: 0.031, ndcg: 0.028, hitRatio: 0.145, trainTime: 0.12 },
  { name: 'Popularity (average)', precision: 0.018, recall: 0.022, ndcg: 0.021, hitRatio: 0.112, trainTime: 0.11 },
  { name: 'UserCF (k=20, cosine)', precision: 0.042, recall: 0.051, ndcg: 0.048, hitRatio: 0.225, trainTime: 2.45 },
  { name: 'ItemCF (k=20, cosine)', precision: 0.048, recall: 0.058, ndcg: 0.053, hitRatio: 0.238, trainTime: 3.78 },
  { name: 'SVD (factors=50)', precision: 0.051, recall: 0.062, ndcg: 0.058, hitRatio: 0.245, trainTime: 42.32 },
  { name: 'NeuralCF (factors=16)', precision: 0.054, recall: 0.065, ndcg: 0.061, hitRatio: 0.252, trainTime: 128.45 },
  { name: 'Hybrid', precision: 0.058, recall: 0.069, ndcg: 0.064, hitRatio: 0.268, trainTime: 48.65 },
];

const userSegmentData = [
  { name: 'Low Activity', precision: 0.042, recall: 0.048, ndcg: 0.045, hitRatio: 0.215 },
  { name: 'Medium Activity', precision: 0.057, recall: 0.068, ndcg: 0.062, hitRatio: 0.265 },
  { name: 'High Activity', precision: 0.069, recall: 0.082, ndcg: 0.075, hitRatio: 0.312 },
];

const genreDistribution = [
  { name: 'Action', value: 242 },
  { name: 'Comedy', value: 198 },
  { name: 'Drama', value: 305 },
  { name: 'Sci-Fi', value: 126 },
  { name: 'Romance', value: 157 },
  { name: 'Thriller', value: 172 },
];

const timelineData = [
  { month: 'Jan', userGrowth: 100, ratingActivity: 450, avgAccuracy: 0.041 },
  { month: 'Feb', userGrowth: 120, ratingActivity: 520, avgAccuracy: 0.044 },
  { month: 'Mar', userGrowth: 150, ratingActivity: 580, avgAccuracy: 0.047 },
  { month: 'Apr', userGrowth: 180, ratingActivity: 620, avgAccuracy: 0.049 },
  { month: 'May', userGrowth: 230, ratingActivity: 700, avgAccuracy: 0.051 },
  { month: 'Jun', userGrowth: 250, ratingActivity: 750, avgAccuracy: 0.054 },
  { month: 'Jul', userGrowth: 280, ratingActivity: 800, avgAccuracy: 0.056 },
  { month: 'Aug', userGrowth: 310, ratingActivity: 820, avgAccuracy: 0.057 },
  { month: 'Sep', userGrowth: 340, ratingActivity: 900, avgAccuracy: 0.058 },
];

const recommendationExamples = [
  {
    userId: 101,
    userProfile: { ageGroup: "25-34", gender: "F", activityLevel: 8 },
    topRated: [
      { title: "The Hidden Dream", releaseYear: 2019, genre: "Sci-Fi|Drama", rating: 5.0 },
      { title: "Eternal Shadow", releaseYear: 2017, genre: "Mystery|Thriller", rating: 4.5 },
      { title: "The Golden Kingdom", releaseYear: 2021, genre: "Fantasy|Adventure", rating: 4.5 },
    ],
    recommendations: [
      { title: "The Silent Legacy", releaseYear: 2020, genre: "Drama|Mystery", score: 4.8 },
      { title: "Rising Star", releaseYear: 2022, genre: "Sci-Fi|Drama", score: 4.7 },
      { title: "Dark Ocean", releaseYear: 2019, genre: "Mystery|Thriller", score: 4.6 },
    ]
  },
  {
    userId: 205,
    userProfile: { ageGroup: "35-44", gender: "M", activityLevel: 6 },
    topRated: [
      { title: "Ancient Empire", releaseYear: 2018, genre: "Action|Adventure", rating: 5.0 },
      { title: "Frozen Heart", releaseYear: 2020, genre: "Drama|Romance", rating: 4.5 },
      { title: "Savage Future", releaseYear: 2021, genre: "Action|Sci-Fi", rating: 4.5 },
    ],
    recommendations: [
      { title: "The Loud Knight", releaseYear: 2022, genre: "Action|Fantasy", score: 4.9 },
      { title: "Golden Dawn", releaseYear: 2019, genre: "Adventure|Drama", score: 4.7 },
      { title: "Hidden Empire", releaseYear: 2021, genre: "Action|Adventure", score: 4.6 },
    ]
  }
];

// Colors for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

const RecommendationDashboard = () => {
  const [activeMetric, setActiveMetric] = useState('precision');
  const [selectedModel, setSelectedModel] = useState(null);
  const [activeTab, setActiveTab] = useState('performance');
  const [selectedUserExample, setSelectedUserExample] = useState(0);

  const handleModelClick = (data) => {
    setSelectedModel(data.name === selectedModel ? null : data.name);
  };

  const filteredData = selectedModel 
    ? modelPerformance.filter(model => model.name === selectedModel)
    : modelPerformance;

  // Format data for radar chart
  const radarData = [
    { metric: 'Precision', hybrid: modelPerformance[6].precision, itemcf: modelPerformance[3].precision, usercf: modelPerformance[2].precision },
    { metric: 'Recall', hybrid: modelPerformance[6].recall, itemcf: modelPerformance[3].recall, usercf: modelPerformance[2].recall },
    { metric: 'NDCG', hybrid: modelPerformance[6].ndcg, itemcf: modelPerformance[3].ndcg, usercf: modelPerformance[2].ndcg },
    { metric: 'Hit Ratio', hybrid: modelPerformance[6].hitRatio, itemcf: modelPerformance[3].hitRatio, usercf: modelPerformance[2].hitRatio },
  ];

  // Currently selected user example
  const userExample = recommendationExamples[selectedUserExample];

  return (
    <div className="p-6 max-w-6xl mx-auto bg-gray-50 rounded-lg shadow">
      <h1 className="text-3xl font-bold mb-2 text-gray-800">Recommendation System Dashboard</h1>
      <p className="text-gray-600 mb-6">Interactive visualization of movie recommendation system performance</p>
      
      {/* Tab navigation */}
      <div className="flex border-b border-gray-200 mb-6">
        <button 
          className={`py-2 px-4 font-medium ${activeTab === 'performance' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTab('performance')}
        >
          Model Performance
        </button>
        <button 
          className={`py-2 px-4 font-medium ${activeTab === 'users' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTab('users')}
        >
          User Segments
        </button>
        <button 
          className={`py-2 px-4 font-medium ${activeTab === 'examples' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTab('examples')}
        >
          Recommendation Examples
        </button>
        <button 
          className={`py-2 px-4 font-medium ${activeTab === 'insights' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTab('insights')}
        >
          Insights
        </button>
      </div>

      {/* Performance tab */}
      {activeTab === 'performance' && (
        <div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">Model Performance</h2>
              <div className="mb-2 flex flex-wrap gap-2">
                <button 
                  onClick={() => setActiveMetric('precision')}
                  className={`px-2 py-1 rounded text-sm ${activeMetric === 'precision' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                >
                  Precision
                </button>
                <button 
                  onClick={() => setActiveMetric('recall')}
                  className={`px-2 py-1 rounded text-sm ${activeMetric === 'recall' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                >
                  Recall
                </button>
                <button 
                  onClick={() => setActiveMetric('ndcg')}
                  className={`px-2 py-1 rounded text-sm ${activeMetric === 'ndcg' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                >
                  NDCG
                </button>
                <button 
                  onClick={() => setActiveMetric('hitRatio')}
                  className={`px-2 py-1 rounded text-sm ${activeMetric === 'hitRatio' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                >
                  Hit Ratio
                </button>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={filteredData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 70 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey={activeMetric} fill="#8884d8" onClick={handleModelClick}>
                    {modelPerformance.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.name === selectedModel ? '#FF8042' : COLORS[index % COLORS.length]} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-gray-500 mt-1 text-center">Click on a bar to filter</div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">Training Time Comparison</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={filteredData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 70 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" />
                  <YAxis label={{ value: 'Seconds', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="trainTime" fill="#82ca9d" onClick={handleModelClick}>
                    {modelPerformance.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.name === selectedModel ? '#FF8042' : COLORS[index % COLORS.length]} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow mb-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">Top Models Comparison</h2>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart outerRadius={150} width={500} height={400} data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis angle={30} domain={[0, 0.35]} />
                <Radar name="Hybrid" dataKey="hybrid" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                <Radar name="Item-CF" dataKey="itemcf" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                <Radar name="User-CF" dataKey="usercf" stroke="#ffc658" fill="#ffc658" fillOpacity={0.6} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* User Segments tab */}
      {activeTab === 'users' && (
        <div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">Performance by User Segment</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={userSegmentData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="precision" name="Precision" fill="#0088FE" />
                  <Bar dataKey="recall" name="Recall" fill="#00C49F" />
                  <Bar dataKey="ndcg" name="NDCG" fill="#FFBB28" />
                  <Bar dataKey="hitRatio" name="Hit Ratio" fill="#FF8042" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">Movie Genre Distribution</h2>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={genreDistribution}
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {genreDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${value} movies`, 'Count']} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow mb-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">System Activity Timeline</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={timelineData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="userGrowth" name="New Users" stroke="#8884d8" activeDot={{ r: 8 }} />
                <Line yAxisId="left" type="monotone" dataKey="ratingActivity" name="Ratings" stroke="#82ca9d" />
                <Line yAxisId="right" type="monotone" dataKey="avgAccuracy" name="Accuracy" stroke="#ffc658" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Recommendation Examples tab */}
      {activeTab === 'examples' && (
        <div>
          <div className="flex space-x-2 mb-4">
            {recommendationExamples.map((example, idx) => (
              <button
                key={example.userId}
                onClick={() => setSelectedUserExample(idx)}
                className={`px-3 py-1 rounded ${selectedUserExample === idx ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800 hover:bg-gray-300'}`}
              >
                User {example.userId}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">User Profile</h2>
              <div className="border-l-4 border-blue-500 pl-3 mb-4">
                <p className="text-sm text-gray-600">User ID</p>
                <p className="font-bold text-lg">{userExample.userId}</p>
              </div>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-600">Age Group</p>
                  <p className="font-medium">{userExample.userProfile.ageGroup}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Gender</p>
                  <p className="font-medium">{userExample.userProfile.gender}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Activity Level</p>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full" 
                      style={{ width: `${userExample.userProfile.activityLevel * 10}%` }}
                    ></div>
                  </div>
                  <p className="text-right text-xs text-gray-500">{userExample.userProfile.activityLevel}/10</p>
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">Top Rated Movies</h2>
              <div className="space-y-4">
                {userExample.topRated.map((movie, idx) => (
                  <div key={idx} className="border-b pb-3">
                    <div className="flex justify-between">
                      <h3 className="font-medium">{movie.title}</h3>
                      <span className="bg-yellow-100 text-yellow-800 font-medium px-2 rounded">★ {movie.rating}</span>
                    </div>
                    <p className="text-sm text-gray-600">{movie.releaseYear} • {movie.genre.replace('|', ', ')}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">Recommendations</h2>
              <div className="space-y-4">
                {userExample.recommendations.map((movie, idx) => (
                  <div key={idx} className="border-b pb-3">
                    <div className="flex justify-between">
                      <h3 className="font-medium">{movie.title}</h3>
                      <span className="bg-green-100 text-green-800 font-medium px-2 rounded">{movie.score.toFixed(1)}</span>
                    </div>
                    <p className="text-sm text-gray-600">{movie.releaseYear} • {movie.genre.replace('|', ', ')}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow mb-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">Recommendation Explanation</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium text-lg mb-2">Why these recommendations?</h3>
                <ul className="list-disc pl-5 space-y-2">
                  <li>
                    <span className="font-medium">Genre Matching:</span> 
                    <span className="text-gray-700"> Based on user's preference for {userExample.topRated[0].genre.split('|')[0]} and {userExample.topRated[0].genre.split('|')[1]} movies</span>
                  </li>
                  <li>
                    <span className="font-medium">Similar Users:</span> 
                    <span className="text-gray-700"> 42 users with similar taste patterns rated these movies highly</span>
                  </li>
                  <li>
                    <span className="font-medium">Recency:</span> 
                    <span className="text-gray-700"> User prefers newer releases ({userExample.topRated.map(m => m.releaseYear).join(", ")})</span>
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium text-lg mb-2">Model Contribution</h3>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm mb-1">Item-based CF</p>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{ width: '65%' }}></div>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm mb-1">User-based CF</p>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: '20%' }}></div>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm mb-1">Matrix Factorization</p>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-purple-500 h-2 rounded-full" style={{ width: '15%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Insights tab */}
      {activeTab === 'insights' && (
        <div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="bg-white p-4 rounded-lg shadow border-l-4 border-blue-500">
              <h2 className="text-xl font-semibold mb-3 text-gray-700">Hybrid Model Superiority</h2>
              <p className="text-gray-700 mb-2">
                The hybrid approach combining collaborative filtering with matrix factorization shows 22% better performance than baseline models.
              </p>
              <div className="flex items-center text-sm mt-2">
                <span className="flex items-center">
                  <span className="h-2 w-2 rounded-full bg-green-500 mr-1"></span>
                  <span>22% increase in precision</span>
                </span>
                <span className="flex items-center ml-4">
                  <span className="h-2 w-2 rounded-full bg-blue-500 mr-1"></span>
                  <span>18% increase in recall</span>
                </span>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow border-l-4 border-green-500">
              <h2 className="text-xl font-semibold mb-3 text-gray-700">Active User Advantage</h2>
              <p className="text-gray-700 mb-2">
                High-activity users receive 39% more accurate recommendations than low-activity users, highlighting the importance of collecting sufficient user data.
              </p>
              <div className="mt-2 grid grid-cols-3 gap-2 text-center text-sm">
                <div className="bg-red-100 p-1 rounded">
                  <div className="font-medium">Low</div>
                  <div>4.2%</div>
                </div>
                <div className="bg-yellow-100 p-1 rounded">
                  <div className="font-medium">Medium</div>
                  <div>5.7%</div>
                </div>
                <div className="bg-green-100 p-1 rounded">
                  <div className="font-medium">High</div>
                  <div>6.9%</div>
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow border-l-4 border-purple-500">
              <h2 className="text-xl font-semibold mb-3 text-gray-700">Genre Influence</h2>
              <p className="text-gray-700 mb-2">
                Drama and Action genres show the strongest collaborative filtering signals, while niche genres benefit more from content-based approaches.
              </p>
              <div className="mt-2">
                <div className="text-sm text-gray-600 mb-1">Most predictable genres:</div>
                <div className="flex flex-wrap gap-1">
                  <span className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded">Drama</span>
                  <span className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded">Action</span>
                  <span className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded">Comedy</span>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">Cold Start Strategy</h2>
              <p className="text-gray-700 mb-3">
                For new users with fewer than 5 ratings, a hybrid approach combining popularity and content-based filtering yields 27% better results than pure collaborative filtering.
              </p>
              <div className="bg-gray-100 p-3 rounded">
                <h3 className="font-medium mb-2">Recommended Strategy:</h3>
                <ol className="list-decimal pl-5 space-y-1">
                  <li>Start with popularity-based recommendations</li>
                  <li>Incorporate content matching after first rating</li>
                  <li>Gradually increase collaborative weight with more ratings</li>
                  <li>Full hybrid model after 5+ ratings</li>
                </ol>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">Training Efficiency</h2>
              <p className="text-gray-700 mb-3">
                Neural CF delivers only 6% performance improvement over Matrix Factorization but requires 3x more training time. For most applications, the simpler model provides better ROI.
              </p>
              <div className="grid grid-cols-2 gap-3 mt-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">3x</div>
                  <div className="text-sm text-gray-600">Training Time Increase</div>
                </div>
                <div className="text-center">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">6%</div>
                        <div className="text-sm text-gray-600">Performance Gain</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
      
            <div className="bg-blue-50 p-4 rounded-lg mt-6">
              <h2 className="text-lg font-semibold mb-2 text-blue-800">Project Summary</h2>
              <p className="text-gray-800">
                Our recommendation system implementation demonstrates that hybrid models combining collaborative filtering and matrix factorization provide the best balance of performance and training efficiency. 
                The system adapts to user activity levels and implements a specialized strategy for cold starts.
              </p>
            </div>
          </div>
        );
      };
      
      export default RecommendationDashboard;
