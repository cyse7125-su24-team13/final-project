FROM caddy:alpine

# Copy your site files to the Caddy default directory
COPY . /usr/share/caddy

# Copy the Caddyfile to the root
COPY Caddyfile /etc/caddy/Caddyfile

# Expose the port that Caddy serves on
EXPOSE 80
