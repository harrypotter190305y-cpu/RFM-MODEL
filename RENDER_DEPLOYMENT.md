# Deploy RFM Model on Render

## Step 1: Prepare Your Project (DONE ✅)
Your project is already on GitHub at:
https://github.com/harrypotter190305y-cpu/RFM-MODEL

## Step 2: Create a Render Account
1. Go to: https://render.com
2. Click "Sign up"
3. Use GitHub to sign up (recommended) or email
4. Authorize Render to access your GitHub repositories

## Step 3: Create a New Web Service
1. Go to: https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Click "Connect" next to your RFM-MODEL repository
4. Or paste: `https://github.com/harrypotter190305y-cpu/RFM-MODEL`

## Step 4: Configure the Service

Fill in these settings:

| Field | Value |
|-------|-------|
| **Name** | `rfm-model` (or any name) |
| **Environment** | `Python 3` |
| **Region** | `Singapore` (or closest to you) |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python app.py` |

## Step 5: Add Environment Variables (Optional)
If you need any env vars (like API keys), click "Add Environment Variable"

## Step 6: Choose Plan
- **Free Plan**: Good for testing (sleeps after 15 min inactivity)
- **Paid Plan**: $7+/month (always running)

## Step 7: Deploy!
Click "Create Web Service" and wait 2-3 minutes

Once deployed, you'll get a URL like:
```
https://rfm-model.onrender.com
```

## Step 8: Access Your App
Open: `https://rfm-model.onrender.com`

Your dashboard will be live at:
- Home: `https://rfm-model.onrender.com/`
- Dataset: `https://rfm-model.onrender.com/dataset`
- Dashboard: `https://rfm-model.onrender.com/dashboard`

---

## Important Notes:

### 1. **requirements.txt**
Make sure `requirements.txt` exists and has all dependencies:
```
Flask==2.3.0
pandas
numpy
matplotlib
scikit-learn
openpyxl
```

### 2. **app.py Port**
Your `app.py` should use the PORT environment variable:
```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')
```

### 3. **Data Files**
Make sure these exist in your repo:
- `data/Online Retail.xlsx` (uploaded to GitHub ✅)
- `templates/` folder (HTML files ✅)
- `artifacts/` folder (images & pickles ✅)

### 4. **View Logs**
In Render dashboard:
- Click your service
- Go to "Logs" tab
- See live output & errors

### 5. **Troubleshooting**
If deployment fails:
1. Check "Logs" for error messages
2. Ensure `requirements.txt` has all packages
3. Check that `app.py` listens on `0.0.0.0` (not `localhost`)
4. Try redeploying: Click "Manual Deploy" → "Deploy latest commit"

---

## TL;DR (Quick Steps)
1. Go to https://render.com
2. Sign up with GitHub
3. New Web Service → Connect RFM-MODEL repo
4. Build: `pip install -r requirements.txt`
5. Start: `python app.py`
6. Click Deploy
7. Wait 2-3 min, open your URL ✅

Done! Your app is live! 🚀
