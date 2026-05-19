const path = require('node:path');
const dotenv = require('dotenv');

dotenv.config({ path: path.resolve(__dirname, '..', '..', '.env') });
dotenv.config();

function required(name) {
  const value = process.env[name];
  if (!value || value.trim() === '') {
    throw new Error(`${name} is required`);
  }
  return value.trim();
}

module.exports = {
  discordToken: required('DISCORD_TOKEN'),
  clientId: required('CLIENT_ID'),
  guildId: required('GUILD_ID'),
  reviewChannelId: required('PNN_REVIEW_CHANNEL_ID'),
  newsChannelId: required('PNN_NEWS_CHANNEL_ID'),
  editorRoleId: required('PNN_EDITOR_ROLE_ID'),
  timeZone: (process.env.PNN_TIME_ZONE || 'Europe/Warsaw').trim(),
  port: Number(process.env.PORT || process.env.PNN_BOT_PORT || 3000),
};
