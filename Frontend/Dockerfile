FROM node:17-alpine

WORKDIR /Frontend

COPY package.json .

RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "start"]