# Use an official Python runtime as a parent image
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /usr/src/app



# Copy the current directory contents into the container at /usr/src/app
COPY . .

COPY requirements.txt /usr/src/app/
RUN pip install -r requirements.txt


# Expose the port that Streamlit runs on
EXPOSE 8501

# Run main.py when the container launches
CMD ["python", "./run_app.py"]
 